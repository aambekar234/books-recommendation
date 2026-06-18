"""Contextual bandit recommender powered by Vowpal Wabbit.

The Multi-Armed Bandit policies in :mod:`mab_books.bandits` are *context-free*:
they learn a single "best book" averaged over everyone. A real recommender,
though, should adapt to *who is asking* — a reader who loves fast-paced
thrillers and one who wants a relaxing fantasy should see different books.

This module models that with a **contextual bandit**. Each round the recommender
is given a :class:`UserContext` (the features describing the current reader) and
must pick a book. Vowpal Wabbit's ``--cb_explore_adf`` learner is used: it
predicts a probability distribution over the books, we sample one to recommend
(this is the *exploration* / experiment part), and after observing the reader's
feedback we teach VW with the realised cost. Quadratic ``-q UA`` interactions
let VW associate user features with book features, so recommendations become
*personalised* to the context rather than globally fixed.

Vowpal Wabbit is an optional dependency. If it is not installed the rest of the
app keeps working; constructing a recommender raises a clear error and callers
can check :data:`VW_AVAILABLE` first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .books import Book

try:  # Vowpal Wabbit ships native wheels but may be absent in some envs.
    import vowpalwabbit

    VW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the dependency
    vowpalwabbit = None
    VW_AVAILABLE = False


# Selectable values surfaced in the UI. "any" means the reader did not specify.
GENRE_OPTIONS = ["any", "science_fiction", "fantasy", "mystery", "programming"]
MOOD_OPTIONS = ["any", "relaxed", "adventurous", "thoughtful", "thrilling"]
READING_LEVEL_OPTIONS = ["any", "casual", "regular", "avid"]


def _clean(value: str) -> str:
    """Sanitise a feature token: VW features cannot contain spaces, ``|`` or ``:``."""
    return re.sub(r"[\s|:]+", "_", str(value).strip().lower()) or "na"


@dataclass(frozen=True)
class UserContext:
    """Features describing the reader the recommendation is for.

    These map to the VW ``User`` namespace and are interacted with each book's
    ``Action`` features so the model can learn *who likes what*.
    """

    pref_genre: str = "any"
    mood: str = "any"
    reading_level: str = "any"

    def to_features(self) -> str:
        """Render as a VW feature string for the ``User`` namespace."""
        return (
            f"pref_genre={_clean(self.pref_genre)} "
            f"mood={_clean(self.mood)} "
            f"level={_clean(self.reading_level)}"
        )

    @property
    def label(self) -> str:
        return f"genre={self.pref_genre}, mood={self.mood}, level={self.reading_level}"


@dataclass
class Recommendation:
    """A single recommendation: the chosen book plus VW's action distribution."""

    action: int
    probabilities: np.ndarray
    book: Book

    @property
    def prob(self) -> float:
        """Probability VW assigned to the action that was actually shown."""
        return float(self.probabilities[self.action])


class VowpalWabbitRecommender:
    """Contextual-bandit book recommender built on ``--cb_explore_adf``.

    Parameters
    ----------
    books:
        Catalogue of books; each one is an action VW can recommend.
    exploration:
        ``"epsilon"`` for epsilon-greedy exploration or ``"softmax"`` to sample
        proportionally to the learned scores.
    epsilon:
        Exploration rate for epsilon-greedy (probability mass spread over all
        actions). Ignored for softmax.
    softmax_lambda:
        Temperature for softmax exploration; higher exploits more.
    interactions:
        When True (default) add ``-q UA`` so user features interact with book
        features — this is what makes recommendations context-dependent.
    rng:
        Optional numpy generator used to *sample* which book to show from VW's
        distribution, for reproducible runs.
    """

    name = "vowpal-wabbit"

    def __init__(
        self,
        books: list[Book],
        *,
        exploration: str = "epsilon",
        epsilon: float = 0.2,
        softmax_lambda: float = 4.0,
        interactions: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not VW_AVAILABLE:
            raise RuntimeError(
                "Vowpal Wabbit is not installed. Install it with "
                "`poetry install` (or `pip install vowpalwabbit`) to use the "
                "contextual recommender."
            )
        if not books:
            raise ValueError("books must not be empty")
        if exploration not in ("epsilon", "softmax"):
            raise ValueError("exploration must be 'epsilon' or 'softmax'")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")

        self.books = books
        self.n_arms = len(books)
        self.exploration = exploration
        self.epsilon = epsilon
        self.softmax_lambda = softmax_lambda
        self.interactions = interactions
        self.rng = rng if rng is not None else np.random.default_rng()

        # Per-book bookkeeping for the UI / metrics (VW holds the real model).
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.rewards = np.zeros(self.n_arms, dtype=np.float64)
        self.total_pulls = 0

        self._workspace = vowpalwabbit.Workspace(self._vw_args())

    # ------------------------------------------------------------------ #
    # VW plumbing
    # ------------------------------------------------------------------ #
    def _vw_args(self) -> str:
        args = ["--cb_explore_adf", "--quiet"]
        if self.exploration == "epsilon":
            args += ["--epsilon", str(self.epsilon)]
        else:
            args += ["--softmax", "--lambda", str(self.softmax_lambda)]
        if self.interactions:
            args += ["-q", "UA"]
        return " ".join(args)

    def _action_features(self, book: Book) -> str:
        return f"genre={_clean(book.subject)} title={_clean(book.title)}"

    def _example(self, context: UserContext, label: tuple[int, float, float] | None = None) -> str:
        """Build a multi-line VW ADF example for the given context.

        ``label`` is ``(action, cost, probability)`` for the chosen book when
        learning, or ``None`` when predicting.
        """
        lines = [f"shared |User {context.to_features()}"]
        for i, book in enumerate(self.books):
            prefix = ""
            if label is not None and label[0] == i:
                # Chosen action line carries `action:cost:probability`; the line
                # position identifies the action so the leading index stays 0.
                prefix = f"0:{label[1]}:{label[2]} "
            lines.append(f"{prefix}|Action {self._action_features(book)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def action_probabilities(self, context: UserContext) -> np.ndarray:
        """Return VW's probability distribution over books for ``context``."""
        probs = self._workspace.predict(self._example(context))
        return np.asarray(probs, dtype=np.float64)

    def recommend(self, context: UserContext) -> Recommendation:
        """Pick a book for ``context`` by sampling VW's action distribution."""
        probs = self.action_probabilities(context)
        total = probs.sum()
        # VW normalises already, but guard against tiny float drift.
        norm = probs / total if total > 0 else np.full(self.n_arms, 1.0 / self.n_arms)
        action = int(self.rng.choice(self.n_arms, p=norm))
        return Recommendation(action=action, probabilities=probs, book=self.books[action])

    def update(self, context: UserContext, action: int, reward: float, prob: float) -> None:
        """Teach VW the outcome of showing ``action`` for ``context``.

        ``reward`` is in [0, 1] (1 = the reader liked it). VW minimises *cost*,
        so we feed it ``cost = -reward``. ``prob`` is the probability VW gave the
        shown action, needed for unbiased off-policy learning.
        """
        if not 0 <= action < self.n_arms:
            raise IndexError(f"action {action} out of range for {self.n_arms} books")
        cost = -float(reward)
        prob = float(min(max(prob, 1e-6), 1.0))  # avoid zero-probability blow-ups
        self._workspace.learn(self._example(context, label=(action, cost, prob)))
        self.counts[action] += 1
        self.rewards[action] += float(reward)
        self.total_pulls += 1

    @property
    def estimated_values(self) -> np.ndarray:
        """Mean observed reward per book (0 for books never shown)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.counts > 0, self.rewards / np.maximum(self.counts, 1), 0.0)

    def reset(self) -> None:
        """Throw away the learned model and all statistics."""
        self.counts[:] = 0
        self.rewards[:] = 0.0
        self.total_pulls = 0
        self._workspace = vowpalwabbit.Workspace(self._vw_args())
