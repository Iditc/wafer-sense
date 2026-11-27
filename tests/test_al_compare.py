"""Tests of the day 8 active-learning comparison."""

import numpy as np
import pytest

from src.active import compare as cmp
from src.active import simulate as sim
from src.active.strategies import STRATEGIES
from tests.test_simulate import toy_data


def fake_history(f1s: list[float], start: int = 100, step: int = 50) -> list[dict]:
    """A curve with the given F1 per round."""
    return [
        {"n_labels": start + step * r, "f1_macro": f, "missed_defect_rate": 0.1}
        for r, f in enumerate(f1s)
    ]


def test_labels_to_reach_first_hit():
    """The first round at or above the target counts."""
    assert cmp.labels_to_reach(fake_history([0.6, 0.75, 0.8])) == 150
    assert cmp.labels_to_reach(fake_history([0.6, 0.7])) is None


def test_random_has_three_seeds_others_one():
    """Only random selection is repeated over seeds."""
    pairs = cmp.runs_to_do()
    assert [s for s, _ in pairs].count("random") == len(cmp.RANDOM_SEEDS)
    assert all(seed == 0 for s, seed in pairs if s != "random")


def test_summarize_band_contains_mean():
    """Mean lies between min and max for every budget."""
    table = cmp.summarize(
        {"random": [fake_history([0.6, 0.7]), fake_history([0.62, 0.74])]}
    )
    row = table["random"]
    assert row["runs"] == 2 and row["labels_to_reach_target"] == [None, None]
    assert all(
        lo <= m <= hi for lo, m, hi in zip(row["f1_min"], row["f1_mean"], row["f1_max"])
    )


@pytest.mark.parametrize("name", [s for s in STRATEGIES if s != "random"])
def test_deterministic_strategies_ignore_the_seed(name):
    """Why entropy, margin and k-center run once: every seed picks the same wafers."""
    x, y = toy_data()
    data = (x[::2], y[::2]), (x[1::2], y[1::2])
    a = sim.simulate(*data, STRATEGIES[name], rounds=2, batch=10, seed=0)
    b = sim.simulate(*data, STRATEGIES[name], rounds=2, batch=10, seed=5)
    assert np.allclose([h["f1_macro"] for h in a], [h["f1_macro"] for h in b])
