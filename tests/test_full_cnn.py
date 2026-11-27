"""Tests of the day 8b full-CNN check of the active-learning label sets."""

import pytest

from src.active import full_cnn as fc

SELECTION = {"start": [0, 1], "rounds": [[5, 6], [3, 4], [9, 8]]}


def test_budget_rows_takes_start_then_whole_rounds():
    """Budget 6 = start set + the first two rounds, in pick order."""
    assert fc.budget_rows("margin", 6, SELECTION).tolist() == [0, 1, 5, 6, 3, 4]


def test_budget_rows_rejects_a_budget_inside_a_round():
    """A budget that splits a round would mix two selections: refused."""
    with pytest.raises(ValueError):
        fc.budget_rows("margin", 5, SELECTION)


def test_every_budget_is_a_round_boundary():
    """All planned budgets are 1,210 + k x 1,210, the simulation's round sizes."""
    assert all((b - 1210) % 1210 == 0 for b in fc.BUDGETS)
    assert set(fc.EPOCHS) == set(fc.BUDGETS)


def test_colab_runs_pair_strategies_per_budget():
    """Each budget has one margin and one random run, named after both."""
    lines = [line.split() for line in fc.colab_runs()]
    assert len(lines) == len(fc.BUDGETS) * len(fc.STRATEGIES)
    for name, strategy, budget, epochs, patience in lines:
        assert name == fc.run_name(strategy, int(budget))
        assert (int(epochs), int(patience)) == fc.EPOCHS[int(budget)]


@pytest.mark.parametrize("strategy", fc.STRATEGIES)
def test_saved_selections_have_no_repeats(strategy):
    """The committed label sets never list a wafer twice."""
    rows = fc.budget_rows(strategy, max(fc.BUDGETS))
    assert len(set(rows.tolist())) == len(rows) == max(fc.BUDGETS)
