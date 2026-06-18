"""Streamlit frontend for the Multi-Armed Bandit book-recommendation demo.

Run with:  poetry run streamlit run mab_books/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import streamlit as st

from mab_books.bandits import build_policy
from mab_books.books import Book, fetch_books, fetch_mixed_catalogue
from mab_books.contextual import (
    GENRE_OPTIONS,
    MOOD_OPTIONS,
    READING_LEVEL_OPTIONS,
    VW_AVAILABLE,
    UserContext,
    VowpalWabbitRecommender,
)
from mab_books.contextual_simulation import (
    build_personas,
    run_contextual_simulation,
)
from mab_books.simulation import assign_true_ctrs, run_simulation

# Genres blended into the contextual recommender's catalogue (excludes "any").
CONTEXTUAL_SUBJECTS = [g for g in GENRE_OPTIONS if g != "any"]

st.set_page_config(page_title="MAB Book Recommender", page_icon="📚", layout="wide")


# --------------------------------------------------------------------------- #
# Cached data loaders
#
# Streamlit re-runs this whole script top-to-bottom on *every* widget
# interaction (including each 👍/👎 click). Without caching, every rerun made a
# fresh network call to Open Library for the catalogue *and* re-downloaded every
# cover image — the main reason the app felt slow and images loaded late.
# Caching makes repeat runs effectively instant.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_books(subject: str, n_books: int) -> list[Book]:
    """Fetch books once per (subject, n_books); cached across reruns."""
    return fetch_books(subject, n_books)


@st.cache_data(show_spinner=False)
def load_mixed_catalogue(subjects: tuple[str, ...], per_subject: int) -> list[Book]:
    """Fetch the multi-genre contextual catalogue once; cached across reruns.

    Like :func:`load_books`, this keeps the contextual tab from re-hitting Open
    Library on every 👍/👎 click.
    """
    return fetch_mixed_catalogue(list(subjects), per_subject=per_subject)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_cover(url: str) -> bytes | None:
    """Download a cover image once and cache the bytes.

    Returns ``None`` (so the caller can show a placeholder) if the image can't
    be fetched. Cached for a day so repeat reruns never hit the network.
    """
    try:
        resp = requests.get(url, timeout=4.0, headers={"User-Agent": "mab-books-demo/0.1"})
        resp.raise_for_status()
        return resp.content
    except requests.RequestException:
        return None


def render_cover(book: Book, width: int = 120) -> None:
    """Show a book's cover, falling back to a styled placeholder.

    The first fetch for a given URL shows a spinner so the user gets immediate
    feedback instead of a blank gap; subsequent reruns hit the cache and the
    spinner never appears.
    """
    data = None
    if book.cover_url:
        with st.spinner("Loading cover…"):
            data = load_cover(book.cover_url)
    if data is not None:
        st.image(data, width=width)
        return
    st.markdown(
        f"""
        <div style="width:{width}px;height:{int(width * 1.5)}px;border-radius:8px;
                    background:linear-gradient(135deg,#6366f1,#a855f7);color:white;
                    display:flex;align-items:center;justify-content:center;
                    text-align:center;padding:8px;font-size:0.8rem;font-weight:600;
                    box-shadow:0 2px 8px rgba(0,0,0,0.2);">
            📖<br>{book.title}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar: configure the data and the bandit algorithm
# --------------------------------------------------------------------------- #
def configure_policy_params(algorithm: str) -> dict:
    """Render algorithm-specific hyper-parameter widgets and return their values."""
    params: dict = {}
    if algorithm == "epsilon-greedy":
        params["epsilon"] = st.sidebar.slider(
            "epsilon (exploration rate)", 0.0, 1.0, 0.1, 0.01,
            help="Probability of trying a random book instead of the best-known one.",
        )
    elif algorithm == "ucb1":
        params["c"] = st.sidebar.slider(
            "c (exploration weight)", 0.0, 5.0, 2.0, 0.1,
            help="Higher values explore uncertain books more aggressively.",
        )
    elif algorithm == "thompson-sampling":
        params["alpha"] = st.sidebar.number_input(
            "alpha prior", min_value=0.1, value=1.0, step=0.1,
            help="Beta prior pseudo-count of successes (clicks).",
        )
        params["beta"] = st.sidebar.number_input(
            "beta prior", min_value=0.1, value=1.0, step=0.1,
            help="Beta prior pseudo-count of failures (skips).",
        )
    return params


def sidebar_config() -> dict:
    st.sidebar.header("⚙️ Configuration")

    subject = st.sidebar.text_input(
        "Book subject / genre", value="science_fiction",
        help="Used to fetch books from Open Library (falls back to a built-in list offline).",
    )
    n_books = st.sidebar.slider("Number of books (arms)", 2, 20, 8)

    st.sidebar.subheader("Bandit algorithm")
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["epsilon-greedy", "ucb1", "thompson-sampling"],
    )
    params = configure_policy_params(algorithm)

    seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    return {
        "subject": subject,
        "n_books": n_books,
        "algorithm": algorithm,
        "params": params,
        "seed": int(seed),
    }


# --------------------------------------------------------------------------- #
# Simulation tab
# --------------------------------------------------------------------------- #
def render_simulation(cfg: dict, books: list) -> None:
    st.subheader("🎲 Simulation")
    st.caption(
        "Each book is given a *hidden* true click-through rate. The bandit does "
        "not see these rates — it must learn which books are best by trying them "
        "and observing simulated clicks."
    )

    n_rounds = st.slider("Number of rounds (recommendations)", 50, 5000, 1000, 50)
    run = st.button("Run simulation", type="primary")
    if not run:
        return

    rng = np.random.default_rng(cfg["seed"])
    true_ctrs = assign_true_ctrs(books, rng)
    policy = build_policy(cfg["algorithm"], n_arms=len(books), rng=rng, **cfg["params"])
    result = run_simulation(policy, true_ctrs, n_rounds, rng)

    total_reward = int(np.sum(result.rewards))
    avg_reward = total_reward / n_rounds
    best_idx = int(np.argmax(true_ctrs))
    pulls = np.bincount(result.choices, minlength=len(books))
    best_share = pulls[best_idx] / n_rounds

    c1, c2, c3 = st.columns(3)
    c1.metric("Total clicks", f"{total_reward}")
    c2.metric("Average reward", f"{avg_reward:.3f}")
    c3.metric("% shown the best book", f"{best_share:.0%}")

    # Per-book breakdown: true vs learned value and how often it was chosen.
    df = pd.DataFrame(
        {
            "Book": [b.label for b in books],
            "True CTR": true_ctrs,
            "Estimated value": policy.estimated_values,
            "Times shown": pulls,
        }
    ).sort_values("True CTR", ascending=False)
    df_display = df.copy()
    df_display.loc[df_display["True CTR"] == true_ctrs[best_idx], "Book"] += "  ⭐"
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("**Cumulative reward** (higher is better)")
    st.line_chart(pd.DataFrame({"cumulative reward": result.cumulative_reward}))

    st.markdown(
        "**Cumulative regret** — clicks lost compared to always showing the best "
        "book. A good bandit flattens out as it learns."
    )
    st.line_chart(pd.DataFrame({"cumulative regret": result.cumulative_regret}))

    st.markdown("**How often each book was recommended**")
    st.bar_chart(df.set_index("Book")["Times shown"])


# --------------------------------------------------------------------------- #
# Interactive tab — the user is the environment
# --------------------------------------------------------------------------- #
def _init_interactive(cfg: dict, books: list) -> None:
    rng = np.random.default_rng(cfg["seed"])
    st.session_state.policy = build_policy(
        cfg["algorithm"], n_arms=len(books), rng=rng, **cfg["params"]
    )
    st.session_state.interactive_books = books
    st.session_state.interactive_rounds = 0
    st.session_state.interactive_clicks = 0
    st.session_state.current_arm = None
    st.session_state.interactive_sig = _config_signature(cfg, books)


def _config_signature(cfg: dict, books: list) -> tuple:
    return (cfg["algorithm"], tuple(sorted(cfg["params"].items())),
            cfg["seed"], tuple(b.key for b in books))


def render_interactive(cfg: dict, books: list) -> None:
    st.subheader("🙋 Interactive — you are the user")
    st.caption(
        "The bandit recommends a book. Tell it whether you'd read it. Your feedback "
        "(Interested = reward 1, Skip = reward 0) updates the algorithm live."
    )

    sig = _config_signature(cfg, books)
    if (
        "policy" not in st.session_state
        or st.session_state.get("interactive_sig") != sig
    ):
        _init_interactive(cfg, books)

    if st.button("Reset learning"):
        _init_interactive(cfg, books)

    policy = st.session_state.policy

    if st.session_state.current_arm is None:
        st.session_state.current_arm = policy.select_arm()
    arm = st.session_state.current_arm
    book = books[arm]

    st.markdown(f"### 📖 Recommended: **{book.title}**")
    st.write(f"*by {book.author}*")
    render_cover(book, width=120)

    c1, c2 = st.columns(2)
    feedback = None
    if c1.button("👍 Interested", use_container_width=True):
        feedback = 1.0
    if c2.button("👎 Skip", use_container_width=True):
        feedback = 0.0

    if feedback is not None:
        policy.update(arm, feedback)
        st.session_state.interactive_rounds += 1
        st.session_state.interactive_clicks += int(feedback)
        st.session_state.current_arm = policy.select_arm()
        st.rerun()

    rounds = st.session_state.interactive_rounds
    clicks = st.session_state.interactive_clicks
    m1, m2, m3 = st.columns(3)
    m1.metric("Rounds", rounds)
    m2.metric("Interested", clicks)
    m3.metric("Hit rate", f"{(clicks / rounds):.0%}" if rounds else "—")

    if rounds:
        df = pd.DataFrame(
            {
                "Book": [b.label for b in books],
                "Estimated value": policy.estimated_values,
                "Times shown": policy.counts,
            }
        ).sort_values("Estimated value", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Contextual tab — Vowpal Wabbit recommends based on *who* is asking
# --------------------------------------------------------------------------- #
def _contextual_controls() -> dict:
    """Render exploration controls for the contextual recommender."""
    c1, c2 = st.columns(2)
    exploration = c1.selectbox(
        "Exploration strategy", ["epsilon", "softmax"],
        help="How VW explores: epsilon-greedy spreads ε over all books; "
        "softmax samples proportionally to learned scores.",
    )
    if exploration == "epsilon":
        explore_param = c2.slider("epsilon (exploration rate)", 0.0, 1.0, 0.2, 0.01)
    else:
        explore_param = c2.slider("softmax λ (temperature)", 0.5, 10.0, 4.0, 0.5)
    interactions = st.checkbox(
        "Personalise (interact user × book features)", value=True,
        help="Enables VW '-q UA' so the model can learn which reader likes which "
        "book. Turn off to see context-blind recommendations.",
    )
    return {
        "exploration": exploration,
        "explore_param": explore_param,
        "interactions": interactions,
    }


def _build_recommender(books: list, ctrl: dict, seed: int) -> VowpalWabbitRecommender:
    kwargs: dict = {
        "exploration": ctrl["exploration"],
        "interactions": ctrl["interactions"],
        "rng": np.random.default_rng(seed),
    }
    if ctrl["exploration"] == "epsilon":
        kwargs["epsilon"] = ctrl["explore_param"]
    else:
        kwargs["softmax_lambda"] = ctrl["explore_param"]
    return VowpalWabbitRecommender(books, **kwargs)


def _contextual_signature(books: list, ctrl: dict, seed: int) -> tuple:
    return (tuple(b.key for b in books), tuple(sorted(ctrl.items())), seed)


def render_contextual(cfg: dict) -> None:
    st.subheader("🧠 Contextual recommendation (Vowpal Wabbit)")
    st.caption(
        "Unlike the bandits above, this recommender uses **who you are** — your "
        "preferred genre, mood and reading habits — to decide what to show. It "
        "learns to give *different* readers *different* books."
    )

    if not VW_AVAILABLE:
        st.error(
            "Vowpal Wabbit is not installed. Run `poetry install` "
            "(or `pip install vowpalwabbit`) and reload to enable this tab."
        )
        return

    with st.spinner("Loading a multi-genre catalogue…"):
        books = load_mixed_catalogue(tuple(CONTEXTUAL_SUBJECTS), 4)
    if not books:
        st.error("No books available for the contextual catalogue.")
        return

    ctrl = _contextual_controls()
    sig = _contextual_signature(books, ctrl, cfg["seed"])
    if (
        "ctx_recommender" not in st.session_state
        or st.session_state.get("ctx_sig") != sig
    ):
        st.session_state.ctx_recommender = _build_recommender(books, ctrl, cfg["seed"])
        st.session_state.ctx_books = books
        st.session_state.ctx_sig = sig
        st.session_state.ctx_rec = None
        st.session_state.ctx_rec_context = None
        st.session_state.ctx_need_new = True
        st.session_state.ctx_rounds = 0
        st.session_state.ctx_likes = 0

    recommender = st.session_state.ctx_recommender

    interactive_tab, experiment_tab = st.tabs(["You are the reader", "Experiment"])
    with interactive_tab:
        _render_contextual_interactive(recommender, books)
    with experiment_tab:
        _render_contextual_experiment(recommender, books, cfg["seed"])


def _render_contextual_interactive(recommender, books: list) -> None:
    st.markdown("#### Set your context")
    c1, c2, c3 = st.columns(3)
    pref_genre = c1.selectbox("Preferred genre", GENRE_OPTIONS)
    mood = c2.selectbox("Mood", MOOD_OPTIONS)
    reading_level = c3.selectbox("Reading habit", READING_LEVEL_OPTIONS)
    context = UserContext(pref_genre=pref_genre, mood=mood, reading_level=reading_level)

    if st.button("Reset learning", key="ctx_reset"):
        recommender.reset()
        st.session_state.ctx_rounds = 0
        st.session_state.ctx_likes = 0
        st.session_state.ctx_need_new = True

    # Regenerate a recommendation when the model is fresh, the context changed,
    # or the reader just gave feedback.
    if (
        st.session_state.ctx_rec is None
        or st.session_state.ctx_rec_context != context
        or st.session_state.ctx_need_new
    ):
        st.session_state.ctx_rec = recommender.recommend(context)
        st.session_state.ctx_rec_context = context
        st.session_state.ctx_need_new = False

    rec = st.session_state.ctx_rec
    book = rec.book

    st.markdown(f"### 📖 Recommended: **{book.title}**")
    st.write(f"*by {book.author}*  ·  genre: `{book.subject}`")
    st.caption(f"VW showed this with probability {rec.prob:.0%} for your context.")
    render_cover(book, width=120)

    c1, c2 = st.columns(2)
    feedback = None
    if c1.button("👍 Interested", use_container_width=True, key="ctx_like"):
        feedback = 1.0
    if c2.button("👎 Skip", use_container_width=True, key="ctx_skip"):
        feedback = 0.0

    if feedback is not None:
        recommender.update(context, rec.action, feedback, rec.prob)
        st.session_state.ctx_rounds += 1
        st.session_state.ctx_likes += int(feedback)
        st.session_state.ctx_need_new = True
        st.rerun()

    rounds = st.session_state.ctx_rounds
    likes = st.session_state.ctx_likes
    m1, m2, m3 = st.columns(3)
    m1.metric("Rounds", rounds)
    m2.metric("Interested", likes)
    m3.metric("Hit rate", f"{(likes / rounds):.0%}" if rounds else "—")

    st.markdown("**What VW would recommend for your current context**")
    probs = recommender.action_probabilities(context)
    df = pd.DataFrame(
        {
            "Book": [b.label for b in books],
            "Genre": [b.subject for b in books],
            "Recommend probability": probs,
            "Times shown": recommender.counts,
        }
    ).sort_values("Recommend probability", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_contextual_experiment(recommender, books: list, seed: int) -> None:
    st.caption(
        "Simulate many readers whose tastes depend on their genre preference. A "
        "working contextual bandit learns to match each reader to their genre, so "
        "regret flattens and the match-rate climbs."
    )
    personas = build_personas(books)
    st.write("Reader personas (one per genre): " + ", ".join(
        f"`{p.pref_genre}`" for p in personas
    ))

    n_rounds = st.slider("Number of readers (rounds)", 100, 5000, 1500, 100, key="ctx_rounds_slider")
    if not st.button("Run experiment", type="primary", key="ctx_run"):
        return

    # Run on a throwaway recommender so the interactive model is untouched.
    sim_rec = _build_recommender(
        books,
        {
            "exploration": recommender.exploration,
            "explore_param": recommender.epsilon
            if recommender.exploration == "epsilon"
            else recommender.softmax_lambda,
            "interactions": recommender.interactions,
        },
        seed,
    )
    result = run_contextual_simulation(sim_rec, n_rounds, np.random.default_rng(seed), personas)

    total_reward = int(np.sum(result.rewards))
    c1, c2, c3 = st.columns(3)
    c1.metric("Total clicks", f"{total_reward}")
    c2.metric("Average reward", f"{total_reward / n_rounds:.3f}")
    c3.metric("Genre match rate", f"{result.match_rate:.0%}")

    st.markdown("**Cumulative regret** — flattens as VW learns each reader's taste.")
    st.line_chart(pd.DataFrame({"cumulative regret": result.cumulative_regret}))

    st.markdown("**What VW recommends for each reader persona** (after training)")
    rows = []
    for p in personas:
        probs = sim_rec.action_probabilities(p)
        top = int(np.argmax(probs))
        rows.append(
            {
                "Reader prefers": p.pref_genre,
                "Top recommendation": books[top].label,
                "Its genre": books[top].subject,
                "Probability": f"{probs[top]:.0%}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("📚 Multi-Armed Bandit Book Recommender")
    st.write(
        "A demo of how Multi-Armed Bandit algorithms balance **exploration** and "
        "**exploitation** to learn which books to recommend. Configure the "
        "algorithm in the sidebar. The **Contextual (VW)** tab goes further, using "
        "Vowpal Wabbit to personalise recommendations to each reader's context."
    )

    cfg = sidebar_config()

    with st.spinner("Loading books…"):
        books = load_books(cfg["subject"], cfg["n_books"])

    if not books:
        st.error("No books available. Try a different subject.")
        return

    st.success(f"Loaded {len(books)} books for '{cfg['subject']}' as bandit arms.")
    with st.expander("Show the book catalogue (arms)"):
        st.dataframe(
            pd.DataFrame({"Title": [b.title for b in books],
                          "Author": [b.author for b in books]}),
            use_container_width=True, hide_index=True,
        )

    sim_tab, interactive_tab, contextual_tab = st.tabs(
        ["Simulation", "Interactive", "Contextual (VW)"]
    )
    with sim_tab:
        render_simulation(cfg, books)
    with interactive_tab:
        render_interactive(cfg, books)
    with contextual_tab:
        render_contextual(cfg)


if __name__ == "__main__":
    main()
