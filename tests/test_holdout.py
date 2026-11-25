"""Tests for the leave-one-type-out run list."""

import pytest

from src.data.plot_utils import DEFECTS
from src.openset.holdout import holdout_runs, run_name


def test_run_name_matches_the_existing_scratch_run():
    assert run_name("Scratch") == "cnn_noscratch"
    assert run_name("Edge-Loc") == "cnn_noedgeloc"


def test_run_name_rejects_o_k_and_unknown_types():
    for name in ("none", "Ring"):
        with pytest.raises(ValueError):
            run_name(name)


def test_holdout_runs_cover_the_other_seven_types_once():
    runs = holdout_runs()
    assert sorted(runs.values()) == sorted(d for d in DEFECTS if d != "Scratch")
    assert len(set(runs)) == 7


def test_holdout_runs_can_include_every_type():
    assert len(holdout_runs(skip=())) == len(DEFECTS)
