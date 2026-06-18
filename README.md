# 📚 Multi-Armed Bandit Book Recommender

An interactive demo of how **Multi-Armed Bandit (MAB)** algorithms balance
*exploration* and *exploitation* — applied to recommending books. Each book is
an "arm"; the bandit learns which books to recommend from observed clicks.

Built with **Streamlit** and managed with **Poetry**.

## What's inside

- **Three configurable bandit policies** (`mab_books/bandits.py`):
  - `epsilon-greedy` — explore randomly with probability ε, otherwise exploit.
  - `ucb1` — Upper Confidence Bound; optimism under uncertainty.
  - `thompson-sampling` — Bayesian sampling from per-arm Beta posteriors.
- **Book catalogue** (`mab_books/books.py`) fetched live from the
  [Open Library](https://openlibrary.org) subject API, with a built-in offline
  fallback so the demo always runs.
- **Simulation environment** (`mab_books/simulation.py`) that assigns each book a
  hidden click-through rate so you can watch the bandit learn and measure regret.
- **Streamlit app** (`mab_books/app.py`) with two modes:
  - **Simulation** — run thousands of rounds and chart cumulative reward, regret,
    and how often each book was shown.
  - **Interactive** — *you* are the user: the bandit recommends a book, you click
    👍/👎, and the algorithm updates live.

## Setup

```bash
poetry install
```

## Run the app

```bash
poetry run streamlit run mab_books/app.py
# or, equivalently:
poetry run mab-books
```

Then open the URL Streamlit prints (default http://localhost:8501).

## Configure the algorithm

Everything is set from the sidebar:

- **Subject / genre** and **number of books** (arms).
- **Algorithm** and its hyper-parameters (ε for epsilon-greedy, `c` for UCB1,
  Beta priors `alpha`/`beta` for Thompson Sampling).
- **Random seed** for reproducible runs.

## Run the tests

```bash
poetry run pytest
```
