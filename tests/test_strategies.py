"""Tests of the active-learning selection strategies (day 7)."""

import numpy as np
import pytest

from src.active import strategies as st


class FixedModel:
    """Stand-in model that returns given probabilities row by row."""

    def __init__(self, probs: np.ndarray):
        self.probs = probs

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probabilities of the rows, looked up by the row id stored in column 0."""
        return self.probs[x[:, 0].astype(int)]


def state_for(probs: np.ndarray, n: int, unlabeled=None) -> st.Round:
    """A round whose embeddings are just row ids, so the model can find each row."""
    x = np.arange(len(probs), dtype=float)[:, None]
    rows = np.arange(len(probs)) if unlabeled is None else np.asarray(unlabeled)
    return st.Round(
        x, FixedModel(probs), np.array([], dtype=int), rows, n, np.random.default_rng(0)
    )


PROBS = np.array(
    [
        [0.98, 0.01, 0.01],  # sure
        [0.34, 0.33, 0.33],  # most uncertain: near uniform
        [0.50, 0.495, 0.005],  # smallest margin (0.005)
        [0.80, 0.10, 0.10],
    ]
)


def test_entropy_picks_the_flattest_prediction():
    """The near-uniform row has the highest entropy."""
    assert st.entropy_strategy(state_for(PROBS, 1)).tolist() == [1]


def test_margin_picks_the_closest_top_two():
    """Row 2 has top-two probabilities 0.50 vs 0.495 (row 1: 0.34 vs 0.33)."""
    assert st.margin_strategy(state_for(PROBS, 1)).tolist() == [2]


def test_uncertainty_strategies_skip_labeled_rows():
    """Only unlabeled rows are returned, as row indices of the full train set."""
    pick = st.entropy_strategy(state_for(PROBS, 2, unlabeled=[0, 2, 3]))
    assert set(pick.tolist()) <= {0, 2, 3} and 1 not in pick


def test_kcenter_picks_spread_out_points():
    """With one labeled point, k-center first takes the far cluster, then the middle."""
    x = np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [0.1, 0.99], [0.7, 0.7]])
    state = st.Round(
        x, None, np.array([0]), np.array([1, 2, 3, 4]), 2, np.random.default_rng(0)
    )
    pick = st.kcenter_strategy(state)
    assert pick[0] in (2, 3) and pick[1] == 4


def test_min_distance_matches_brute_force():
    """Blocked distances equal a direct computation."""
    rng = np.random.default_rng(1)
    a, b = st.unit(rng.normal(size=(50, 4))), st.unit(rng.normal(size=(30, 4)))
    direct = np.linalg.norm(a[:, None] - b[None], axis=2).min(axis=1)
    assert np.allclose(
        st.min_distance(a.astype(np.float32), b.astype(np.float32)), direct, atol=1e-3
    )


@pytest.mark.parametrize("name", sorted(st.STRATEGIES))
def test_every_strategy_returns_n_distinct_unlabeled_rows(name):
    """Right count, no repeats, nothing outside the pool."""
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(3), size=40)
    x = np.column_stack([np.arange(40), rng.normal(size=(40, 3))])
    unlabeled = np.arange(10, 40)
    state = st.Round(
        x, FixedModel(probs), np.arange(10), unlabeled, 7, np.random.default_rng(0)
    )
    pick = st.STRATEGIES[name](state)
    assert (
        len(pick) == 7
        and len(set(pick.tolist())) == 7
        and np.isin(pick, unlabeled).all()
    )


def test_random_same_seed_same_pick():
    """Random selection depends only on the seed."""
    a = st.random_strategy(state_for(PROBS, 2))
    b = st.random_strategy(state_for(PROBS, 2))
    assert a.tolist() == b.tolist()
