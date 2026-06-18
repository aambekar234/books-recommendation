"""Synthetic environment for the contextual recommender.

The context-free simulation in :mod:`mab_books.simulation` gives every book a
single hidden click-through rate. Here the reward depends on *both* the reader
and the book: a synthetic reader has a preferred genre, and a book is much more
likely to be clicked when its subject matches that preference.

This lets the demo show the key property of a contextual bandit — that it learns
*different* recommendations for *different* readers — and measures how quickly it
does so via cumulative regret against the best book for each reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .books import Book
from .contextual import UserContext, VowpalWabbitRecommender

# Mood / reading-level values the synthetic readers are drawn from, so the user
# namespace carries more than just the genre preference.
_MOODS = ["relaxed", "adventurous", "thoughtful", "thrilling"]
_LEVELS = ["casual", "regular", "avid"]


def build_personas(books: list[Book]) -> list[UserContext]:
    """One reader persona per genre present in ``books``.

    Mood and reading level are left as ``"any"`` here; the simulation fills them
    with random values each round so the feature space is non-trivial.
    """
    genres = sorted({b.subject for b in books})
    return [UserContext(pref_genre=g) for g in genres]


def context_book_ctr(
    context: UserContext,
    book: Book,
    match: float = 0.85,
    miss: float = 0.1,
) -> float:
    """Hidden true CTR of showing ``book`` to a reader with ``context``.

    High when the book's genre matches the reader's preference, low otherwise.
    """
    return match if book.subject == context.pref_genre else miss


@dataclass
class ContextualSimResult:
    """History captured while running the contextual recommender."""

    contexts: list[UserContext] = field(default_factory=list)
    choices: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    matched: list[bool] = field(default_factory=list)
    regrets: list[float] = field(default_factory=list)

    @property
    def cumulative_reward(self) -> np.ndarray:
        return np.cumsum(self.rewards) if self.rewards else np.array([])

    @property
    def cumulative_regret(self) -> np.ndarray:
        return np.cumsum(self.regrets) if self.regrets else np.array([])

    @property
    def match_rate(self) -> float:
        """Fraction of rounds where the shown book matched the reader's genre."""
        return float(np.mean(self.matched)) if self.matched else 0.0


def run_contextual_simulation(
    recommender: VowpalWabbitRecommender,
    n_rounds: int,
    rng: np.random.Generator,
    personas: list[UserContext] | None = None,
) -> ContextualSimResult:
    """Run ``recommender`` for ``n_rounds`` against synthetic, context-aware readers."""
    books = recommender.books
    if personas is None:
        personas = build_personas(books)
    if not personas:
        raise ValueError("no personas available to simulate")

    # Best achievable CTR per persona — used to compute per-round regret.
    best_ctr = {
        p.pref_genre: max(context_book_ctr(p, b) for b in books) for p in personas
    }

    result = ContextualSimResult()
    for _ in range(n_rounds):
        base = personas[int(rng.integers(len(personas)))]
        context = UserContext(
            pref_genre=base.pref_genre,
            mood=_MOODS[int(rng.integers(len(_MOODS)))],
            reading_level=_LEVELS[int(rng.integers(len(_LEVELS)))],
        )
        rec = recommender.recommend(context)
        ctr = context_book_ctr(context, rec.book)
        reward = 1.0 if rng.random() < ctr else 0.0
        recommender.update(context, rec.action, reward, rec.prob)

        result.contexts.append(context)
        result.choices.append(rec.action)
        result.rewards.append(reward)
        result.matched.append(rec.book.subject == context.pref_genre)
        result.regrets.append(best_ctr[context.pref_genre] - ctr)
    return result
