"""Simulated environment so the bandit has something to learn from.

In a real product the reward would come from a live user clicking (or not
clicking) a recommended book. For a self-contained demo we model each book as
having a hidden true click-through rate (CTR). Each round the chosen book is
"shown" and yields reward 1 with probability equal to its true CTR.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bandits import BanditPolicy
from .books import Book


def assign_true_ctrs(
    books: list[Book],
    rng: np.random.Generator,
    low: float = 0.05,
    high: float = 0.6,
) -> np.ndarray:
    """Assign each book a hidden true CTR drawn uniformly from [low, high]."""
    return rng.uniform(low, high, size=len(books))


@dataclass
class SimulationResult:
    """History captured while running a bandit against the environment."""

    choices: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    true_ctrs: np.ndarray | None = None

    @property
    def cumulative_reward(self) -> np.ndarray:
        return np.cumsum(self.rewards) if self.rewards else np.array([])

    @property
    def cumulative_regret(self) -> np.ndarray:
        """Regret = reward lost vs. always picking the truly best arm."""
        if not self.choices or self.true_ctrs is None:
            return np.array([])
        best = self.true_ctrs.max()
        chosen = self.true_ctrs[np.array(self.choices)]
        return np.cumsum(best - chosen)


def run_simulation(
    policy: BanditPolicy,
    true_ctrs: np.ndarray,
    n_rounds: int,
    rng: np.random.Generator,
) -> SimulationResult:
    """Run ``policy`` for ``n_rounds`` against a Bernoulli reward environment."""
    if len(true_ctrs) != policy.n_arms:
        raise ValueError("true_ctrs length must match number of arms")
    result = SimulationResult(true_ctrs=true_ctrs)
    for _ in range(n_rounds):
        arm = policy.select_arm()
        reward = 1.0 if rng.random() < true_ctrs[arm] else 0.0
        policy.update(arm, reward)
        result.choices.append(arm)
        result.rewards.append(reward)
    return result
