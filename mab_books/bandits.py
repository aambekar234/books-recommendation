"""Multi-Armed Bandit policies.

Each policy treats every book as an "arm". The policy chooses which arm to
pull (which book to recommend) and is updated with a binary reward
(1 = the user clicked / liked, 0 = ignored). The goal is to maximise the
total reward over time by balancing *exploration* (trying arms whose value
is uncertain) against *exploitation* (showing the arm that looks best so far).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


class BanditPolicy(ABC):
    """Common interface for all bandit policies.

    Parameters
    ----------
    n_arms:
        Number of arms (books) the policy chooses between.
    rng:
        Optional numpy random generator for reproducible runs.
    """

    name = "base"

    def __init__(self, n_arms: int, rng: np.random.Generator | None = None) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        self.n_arms = n_arms
        self.rng = rng if rng is not None else np.random.default_rng()
        # counts[i] = number of times arm i was pulled
        # rewards[i] = cumulative reward collected from arm i
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.rewards = np.zeros(n_arms, dtype=np.float64)
        self.total_pulls = 0

    @property
    def estimated_values(self) -> np.ndarray:
        """Mean reward observed per arm (0 for arms never pulled)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(self.counts > 0, self.rewards / np.maximum(self.counts, 1), 0.0)
        return values

    @abstractmethod
    def select_arm(self) -> int:
        """Return the index of the arm to pull next."""

    def update(self, arm: int, reward: float) -> None:
        """Record the observed ``reward`` for pulling ``arm``."""
        if not 0 <= arm < self.n_arms:
            raise IndexError(f"arm {arm} out of range for {self.n_arms} arms")
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.total_pulls += 1

    def reset(self) -> None:
        self.counts[:] = 0
        self.rewards[:] = 0.0
        self.total_pulls = 0


class EpsilonGreedy(BanditPolicy):
    """With probability ``epsilon`` explore a random arm, else exploit the best.

    ``epsilon`` controls the exploration rate (0 = pure greedy).
    """

    name = "epsilon-greedy"

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n_arms, rng)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon

    def select_arm(self) -> int:
        # Pull each arm once before exploiting, so estimates are seeded.
        unseen = np.where(self.counts == 0)[0]
        if unseen.size > 0:
            return int(unseen[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        values = self.estimated_values
        # Break ties randomly among the best arms.
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))


class UCB1(BanditPolicy):
    """Upper Confidence Bound: pick the arm with the highest optimistic estimate.

    Each arm's score is ``mean + c * sqrt(2 * ln(total) / count)``. The bonus
    term shrinks as an arm is pulled more, so rarely-tried arms are favoured
    until proven worse. ``c`` scales how aggressively we explore.
    """

    name = "ucb1"

    def __init__(
        self,
        n_arms: int,
        c: float = 2.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n_arms, rng)
        if c < 0:
            raise ValueError("c must be >= 0")
        self.c = c

    def select_arm(self) -> int:
        unseen = np.where(self.counts == 0)[0]
        if unseen.size > 0:
            return int(unseen[0])
        means = self.estimated_values
        bonus = np.sqrt(self.c * math.log(self.total_pulls) / self.counts)
        scores = means + bonus
        best = np.flatnonzero(scores == scores.max())
        return int(self.rng.choice(best))


class ThompsonSampling(BanditPolicy):
    """Bayesian policy modelling each arm's reward with a Beta distribution.

    Each arm keeps a Beta(alpha, beta) posterior over its success probability.
    To choose, we sample once from every posterior and pull the arm with the
    highest sample, then update the posterior with the observed reward. Reward
    is treated as a Bernoulli outcome in [0, 1].
    """

    name = "thompson-sampling"

    def __init__(
        self,
        n_arms: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n_arms, rng)
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta priors must be > 0")
        self.alpha0 = alpha
        self.beta0 = beta
        self.alpha = np.full(n_arms, alpha, dtype=np.float64)
        self.beta = np.full(n_arms, beta, dtype=np.float64)

    def select_arm(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        best = np.flatnonzero(samples == samples.max())
        return int(self.rng.choice(best))

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        reward = float(np.clip(reward, 0.0, 1.0))
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward

    def reset(self) -> None:
        super().reset()
        self.alpha[:] = self.alpha0
        self.beta[:] = self.beta0


POLICIES = {
    EpsilonGreedy.name: EpsilonGreedy,
    UCB1.name: UCB1,
    ThompsonSampling.name: ThompsonSampling,
}


def build_policy(name: str, n_arms: int, rng=None, **params) -> BanditPolicy:
    """Factory that builds a policy by ``name`` with the given hyper-parameters."""
    if name not in POLICIES:
        raise ValueError(f"unknown policy '{name}'. choose from {list(POLICIES)}")
    return POLICIES[name](n_arms=n_arms, rng=rng, **params)
