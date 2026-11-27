"""Tests of the simulated active-learning loop (day 6)."""

import numpy as np
import pytest

from src.active import simulate as sim
from src.active.strategies import random_strategy
from src.data.plot_utils import CLASSES


def toy_data(n_per_class: int = 60, dim: int = 8, seed: int = 0) -> tuple:
    """Separable embeddings: each class is a blob around its own center."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(len(CLASSES)), n_per_class)
    centers = rng.normal(0, 5, (len(CLASSES), dim))
    x = (centers[y] + rng.normal(0, 1, (len(y), dim))).astype(np.float32)
    return x, y


@pytest.fixture
def data():
    """Train and val toy sets with the same class centers."""
    x, y = toy_data()
    return (x[::2], y[::2]), (x[1::2], y[1::2])


def test_oracle_rejects_a_second_selection():
    """A wafer revealed once cannot be revealed again."""
    oracle = sim.Oracle(np.arange(10))
    oracle.reveal(np.array([1, 2]))
    with pytest.raises(ValueError):
        oracle.reveal(np.array([2, 3]))
    with pytest.raises(ValueError):
        oracle.reveal(np.array([4, 4]))


def test_oracle_shows_only_revealed_labels():
    """Labels of unselected wafers stay hidden."""
    oracle = sim.Oracle(np.array([5, 6, 7, 8]))
    oracle.reveal(np.array([2]))
    rows, labels = oracle.labeled()
    assert rows.tolist() == [2] and labels.tolist() == [7]


def test_budget_and_no_repeats(data):
    """Each round adds exactly `batch` new wafers, never one already labeled."""
    seen = []

    def spy(state):
        pick = random_strategy(state)
        assert len(pick) == state.n
        assert not np.isin(pick, np.concatenate(seen)).any() if seen else True
        seen.append(pick)
        return pick

    history = sim.simulate(*data, spy, rounds=3, batch=20)
    start = history[0]["n_labels"]
    assert [h["n_labels"] for h in history] == [start + 20 * r for r in range(4)]


def test_strategy_sees_only_unlabeled_train_rows(data):
    """Unlabeled rows are train indices that were never revealed (val never enters)."""
    train, _ = data

    def check(state):
        assert state.unlabeled.max() < len(train[1])
        assert state.x is train[0]
        return random_strategy(state)

    sim.simulate(*data, check, rounds=2, batch=10)


def test_same_seed_same_curve(data):
    """The same seed gives the same selections and results."""
    a = sim.simulate(*data, random_strategy, rounds=2, batch=15, seed=3)
    b = sim.simulate(*data, random_strategy, rounds=2, batch=15, seed=3)
    assert [h["f1_macro"] for h in a] == [h["f1_macro"] for h in b]


def test_start_set_matches_the_learner(data):
    """Round 0 uses the 1% start set cnn_frac01 was trained on, whatever the seed."""
    train, _ = data
    expected = len(sim.sample_labels(train[1], sim.START_FRAC, sim.START_SEED))
    for seed in (0, 1):
        history = sim.simulate(*data, random_strategy, rounds=0, seed=seed)
        assert history[0]["n_labels"] == expected


def test_stops_when_pool_is_empty(data):
    """A batch larger than the pool labels everything and stops."""
    train, _ = data
    history = sim.simulate(*data, random_strategy, rounds=5, batch=10_000)
    assert history[-1]["n_labels"] == len(train[1])
