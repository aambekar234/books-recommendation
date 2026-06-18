import numpy as np
import pytest

from mab_books.bandits import (
    EpsilonGreedy,
    ThompsonSampling,
    UCB1,
    build_policy,
)
from mab_books.simulation import assign_true_ctrs, run_simulation
from mab_books.books import Book


def rng(seed=0):
    return np.random.default_rng(seed)


@pytest.mark.parametrize("name", ["epsilon-greedy", "ucb1", "thompson-sampling"])
def test_select_arm_in_range_and_update(name):
    policy = build_policy(name, n_arms=4, rng=rng())
    for _ in range(50):
        arm = policy.select_arm()
        assert 0 <= arm < 4
        policy.update(arm, 1.0)
    assert policy.total_pulls == 50
    assert policy.counts.sum() == 50


def test_epsilon_greedy_pure_greedy_exploits_best():
    policy = EpsilonGreedy(n_arms=3, epsilon=0.0, rng=rng())
    # Seed each arm once, then make arm 1 clearly best.
    for arm in range(3):
        policy.update(arm, 0.0)
    policy.update(1, 1.0)
    # Pure greedy with arm 1 best should keep choosing arm 1.
    assert policy.select_arm() == 1


def test_epsilon_greedy_validates_epsilon():
    with pytest.raises(ValueError):
        EpsilonGreedy(n_arms=2, epsilon=1.5)


def test_ucb1_pulls_each_arm_before_exploiting():
    policy = UCB1(n_arms=3, rng=rng())
    seen = set()
    for _ in range(3):
        arm = policy.select_arm()
        policy.update(arm, 0.0)
        seen.add(arm)
    # The first three selections must cover every previously-unseen arm.
    assert seen == {0, 1, 2}


def test_thompson_posterior_updates():
    policy = ThompsonSampling(n_arms=2, rng=rng())
    policy.update(0, 1.0)
    policy.update(1, 0.0)
    assert policy.alpha[0] == 2.0  # prior 1 + reward 1
    assert policy.beta[1] == 2.0   # prior 1 + (1 - reward)


def test_reset_clears_state():
    policy = ThompsonSampling(n_arms=2, rng=rng())
    policy.update(0, 1.0)
    policy.reset()
    assert policy.total_pulls == 0
    assert policy.counts.sum() == 0
    assert policy.alpha[0] == 1.0


def test_update_rejects_out_of_range_arm():
    policy = EpsilonGreedy(n_arms=2)
    with pytest.raises(IndexError):
        policy.update(5, 1.0)


def test_build_policy_unknown_name():
    with pytest.raises(ValueError):
        build_policy("does-not-exist", n_arms=2)


def test_simulation_learns_best_arm():
    # The best arm has a much higher CTR; a good bandit should mostly pick it.
    books = [Book(f"k{i}", f"Book {i}", "Author", "test") for i in range(4)]
    r = rng(123)
    true_ctrs = np.array([0.1, 0.1, 0.9, 0.1])
    policy = build_policy("ucb1", n_arms=len(books), rng=r)
    result = run_simulation(policy, true_ctrs, n_rounds=2000, rng=r)

    pulls = np.bincount(result.choices, minlength=4)
    assert pulls.argmax() == 2
    # Regret should grow sub-linearly: average regret per round stays small.
    assert result.cumulative_regret[-1] / 2000 < 0.1


def test_run_simulation_validates_arm_count():
    policy = build_policy("ucb1", n_arms=3, rng=rng())
    with pytest.raises(ValueError):
        run_simulation(policy, np.array([0.5, 0.5]), n_rounds=10, rng=rng())


def test_assign_true_ctrs_bounds():
    books = [Book(f"k{i}", f"B{i}", "A", "test") for i in range(5)]
    ctrs = assign_true_ctrs(books, rng(), low=0.2, high=0.4)
    assert ctrs.shape == (5,)
    assert np.all(ctrs >= 0.2) and np.all(ctrs <= 0.4)
