"""Streamlit frontend for the Multi-Armed Bandit book-recommendation demo.

Run with:  poetry run streamlit run mab_books/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from mab_books.bandits import build_policy
from mab_books.books import fetch_books
from mab_books.simulation import assign_true_ctrs, run_simulation

st.set_page_config(page_title="MAB Book Recommender", page_icon="📚", layout="wide")


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
    if book.cover_url:
        st.image(book.cover_url, width=120)

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
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("📚 Multi-Armed Bandit Book Recommender")
    st.write(
        "A demo of how Multi-Armed Bandit algorithms balance **exploration** and "
        "**exploitation** to learn which books to recommend. Configure the "
        "algorithm in the sidebar."
    )

    cfg = sidebar_config()

    with st.spinner("Loading books…"):
        books = fetch_books(cfg["subject"], cfg["n_books"])

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

    sim_tab, interactive_tab = st.tabs(["Simulation", "Interactive"])
    with sim_tab:
        render_simulation(cfg, books)
    with interactive_tab:
        render_interactive(cfg, books)


if __name__ == "__main__":
    main()
