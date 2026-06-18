import numpy as np
import pytest

from mab_books.books import Book, fetch_mixed_catalogue, FALLBACK_BOOKS
from mab_books.contextual import (
    VW_AVAILABLE,
    UserContext,
    VowpalWabbitRecommender,
    _clean,
)
from mab_books.contextual_simulation import (
    build_personas,
    context_book_ctr,
    run_contextual_simulation,
)

# The whole module needs Vowpal Wabbit; skip cleanly if it is unavailable.
pytestmark = pytest.mark.skipif(not VW_AVAILABLE, reason="vowpalwabbit not installed")


def rng(seed=0):
    return np.random.default_rng(seed)


def sample_books():
    """A multi-genre catalogue drawn from the offline fallback list."""
    subjects = ["science_fiction", "fantasy", "mystery"]
    return [b for b in FALLBACK_BOOKS if b.subject in subjects]


# --------------------------------------------------------------------------- #
# UserContext / feature sanitisation
# --------------------------------------------------------------------------- #
def test_clean_strips_unsafe_characters():
    assert _clean("Science Fiction") == "science_fiction"
    assert _clean("a|b:c") == "a_b_c"
    assert _clean("") == "na"


def test_user_context_features_are_vw_safe():
    ctx = UserContext(pref_genre="science fiction", mood="Relaxed", reading_level="avid")
    feats = ctx.to_features()
    assert "science_fiction" in feats
    assert "|" not in feats and ":" not in feats


# --------------------------------------------------------------------------- #
# Recommender basics
# --------------------------------------------------------------------------- #
def test_recommend_in_range_and_update_tracks_stats():
    books = sample_books()
    rec_engine = VowpalWabbitRecommender(books, rng=rng())
    ctx = UserContext(pref_genre="fantasy")
    for _ in range(20):
        rec = rec_engine.recommend(ctx)
        assert 0 <= rec.action < len(books)
        assert rec.book is books[rec.action]
        rec_engine.update(ctx, rec.action, 1.0, rec.prob)
    assert rec_engine.total_pulls == 20
    assert rec_engine.counts.sum() == 20


def test_probabilities_sum_to_one():
    books = sample_books()
    rec_engine = VowpalWabbitRecommender(books, epsilon=0.3, rng=rng())
    probs = rec_engine.action_probabilities(UserContext(pref_genre="mystery"))
    assert probs.shape == (len(books),)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_update_rejects_out_of_range_action():
    rec_engine = VowpalWabbitRecommender(sample_books(), rng=rng())
    with pytest.raises(IndexError):
        rec_engine.update(UserContext(), 999, 1.0, 0.5)


def test_invalid_exploration_and_epsilon():
    with pytest.raises(ValueError):
        VowpalWabbitRecommender(sample_books(), exploration="bogus")
    with pytest.raises(ValueError):
        VowpalWabbitRecommender(sample_books(), epsilon=1.5)


def test_empty_books_rejected():
    with pytest.raises(ValueError):
        VowpalWabbitRecommender([], rng=rng())


def test_reset_clears_stats():
    rec_engine = VowpalWabbitRecommender(sample_books(), rng=rng())
    ctx = UserContext(pref_genre="fantasy")
    rec = rec_engine.recommend(ctx)
    rec_engine.update(ctx, rec.action, 1.0, rec.prob)
    rec_engine.reset()
    assert rec_engine.total_pulls == 0
    assert rec_engine.counts.sum() == 0


# --------------------------------------------------------------------------- #
# Contextual learning — the core behaviour the issue asks for
# --------------------------------------------------------------------------- #
def test_learns_context_dependent_recommendations():
    """After training, VW should recommend a reader's preferred genre."""
    books = sample_books()
    rec_engine = VowpalWabbitRecommender(books, epsilon=0.1, rng=rng(7))
    run_contextual_simulation(rec_engine, n_rounds=1200, rng=rng(7))

    for genre in ("science_fiction", "fantasy", "mystery"):
        probs = rec_engine.action_probabilities(UserContext(pref_genre=genre))
        top = int(np.argmax(probs))
        assert books[top].subject == genre, f"{genre} reader got {books[top].subject}"


def test_simulation_regret_and_match_rate():
    books = sample_books()
    rec_engine = VowpalWabbitRecommender(books, epsilon=0.1, rng=rng(3))
    result = run_contextual_simulation(rec_engine, n_rounds=1500, rng=rng(3))

    # Average regret per round should be small once VW has learned.
    assert result.cumulative_regret[-1] / 1500 < 0.2
    # Most readers end up shown a book in their preferred genre.
    assert result.match_rate > 0.6


def test_interactions_off_does_not_personalise():
    """Without -q UA, recommendations cannot depend on the reader's genre."""
    books = sample_books()
    rec_engine = VowpalWabbitRecommender(books, epsilon=0.1, interactions=False, rng=rng(5))
    run_contextual_simulation(rec_engine, n_rounds=1200, rng=rng(5))

    tops = {
        genre: int(np.argmax(rec_engine.action_probabilities(UserContext(pref_genre=genre))))
        for genre in ("science_fiction", "fantasy", "mystery")
    }
    # A context-blind model converges to the same global favourite for everyone.
    assert len(set(tops.values())) == 1


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #
def test_build_personas_one_per_genre():
    books = sample_books()
    personas = build_personas(books)
    assert {p.pref_genre for p in personas} == {b.subject for b in books}


def test_context_book_ctr_prefers_matching_genre():
    book = Book("k", "T", "A", "fantasy")
    assert context_book_ctr(UserContext(pref_genre="fantasy"), book) > context_book_ctr(
        UserContext(pref_genre="mystery"), book
    )


def test_run_contextual_simulation_requires_personas():
    rec_engine = VowpalWabbitRecommender(sample_books(), rng=rng())
    with pytest.raises(ValueError):
        run_contextual_simulation(rec_engine, n_rounds=10, rng=rng(), personas=[])


# --------------------------------------------------------------------------- #
# Mixed catalogue helper (does not need VW, but lives with the feature)
# --------------------------------------------------------------------------- #
def test_fetch_mixed_catalogue_dedupes_and_spans_genres(monkeypatch):
    calls = {}

    def fake_fetch(subject, limit):
        calls[subject] = limit
        return [b for b in FALLBACK_BOOKS if b.subject == subject][:limit]

    monkeypatch.setattr("mab_books.books.fetch_books", fake_fetch)
    catalogue = fetch_mixed_catalogue(["science_fiction", "fantasy"], per_subject=2)

    subjects = {b.subject for b in catalogue}
    assert subjects == {"science_fiction", "fantasy"}
    keys = [b.key for b in catalogue]
    assert len(keys) == len(set(keys))  # no duplicates
