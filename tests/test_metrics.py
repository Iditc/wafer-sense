"""Unit tests: metrics checked against small hand-computed examples."""

import numpy as np
import pytest

from src.data.plot_utils import CLASSES
from src.models.ae_eval import evaluate_scores, operating_point
from src.models.metrics import OK_INDEX, defect_vs_ok, evaluate

CENTER, SCRATCH = CLASSES.index("Center"), CLASSES.index("Scratch")


def test_ok_index_is_the_none_class():
    assert CLASSES[OK_INDEX] == "none"


def test_defect_vs_ok_counts_the_two_costly_errors():
    y_true = np.array([0, 0, 0, 0, CENTER, CENTER, SCRATCH, SCRATCH])
    y_pred = np.array([0, 0, 0, CENTER, 0, CENTER, CENTER, SCRATCH])
    result = defect_vs_ok(y_true, y_pred)
    assert result["missed_defect_rate"] == pytest.approx(1 / 4)
    assert result["false_alarm_rate"] == pytest.approx(1 / 4)


def test_wrong_defect_type_is_not_a_missed_defect():
    result = defect_vs_ok(np.array([CENTER]), np.array([SCRATCH]))
    assert result["missed_defect_rate"] == 0


def test_defect_vs_ok_handles_a_single_group():
    assert defect_vs_ok(np.zeros(3, int), np.zeros(3, int))["missed_defect_rate"] == 0
    only_defects = np.full(3, CENTER)
    assert defect_vs_ok(only_defects, only_defects)["false_alarm_rate"] == 0


def test_perfect_predictions():
    y = np.arange(9).repeat(3)
    result = evaluate(y, y)
    assert result["f1_macro"] == 1.0 and result["accuracy"] == 1.0
    assert np.trace(result["confusion_matrix"]) == len(y)


def test_macro_f1_counts_every_class_equally():
    y_true = np.concatenate([np.zeros(90, int), np.arange(1, 9).repeat(10)])
    result = evaluate(y_true, np.zeros_like(y_true))
    ok_f1 = 2 * 90 / (170 + 90)
    assert result["f1_macro"] == pytest.approx(ok_f1 / 9)
    assert result["missed_defect_rate"] == 1.0


def test_confusion_matrix_rows_are_true_classes():
    result = evaluate(np.array([CENTER, CENTER]), np.array([CENTER, SCRATCH]))
    matrix = np.array(result["confusion_matrix"])
    assert matrix.shape == (9, 9)
    assert matrix[CENTER, CENTER] == 1 and matrix[CENTER, SCRATCH] == 1


def test_per_class_uses_display_names():
    result = evaluate(np.arange(9), np.arange(9))
    assert "O.K." in result["per_class"] and "none" not in result["per_class"]


def test_operating_point_matches_hand_count():
    y = np.array([0, 0, 0, 0, CENTER, CENTER, SCRATCH, SCRATCH])
    score = np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.8, 0.7, 0.95])
    result = operating_point(score, y, 0.5)
    assert result["confusion_matrix"] == [[3, 1], [1, 3]]
    assert result["precision"] == pytest.approx(3 / 4)
    assert result["recall"] == pytest.approx(3 / 4)
    assert result["f1"] == pytest.approx(3 / 4)
    assert result["caught_per_class"] == {"Center": 0.5, "Scratch": 1.0}


def test_operating_point_score_on_threshold_is_not_flagged():
    result = operating_point(np.array([0.5, 0.5]), np.array([0, CENTER]), 0.5)
    assert result["confusion_matrix"] == [[1, 0], [1, 0]]


def test_evaluate_scores_picks_the_best_score():
    y = np.array([0] * 100 + [CENTER] * 100)
    rng = np.random.default_rng(0)
    scores = {"good": y / CENTER + 0.01 * rng.random(200), "noise": rng.random(200)}
    result = evaluate_scores(y, scores, false_alarm=0.05, max_missed=0.04)
    assert result["best_score"] == "good"
    assert result["auroc"]["good"] == 1.0
    assert result["at_false_alarm"]["false_alarm_rate"] <= 0.05
    assert result["at_max_missed"]["missed_defect_rate"] <= 0.05
