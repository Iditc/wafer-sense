"""Tests for the day 5 new-type scores."""

import numpy as np
import pytest

from src.models.embed_eval import knn_distance
from src.openset import scores as sc
from src.openset.compare_scores import threshold


def clusters(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Three tight clusters of 16-d embeddings, 50 rows each, labels 0-2."""
    rng = np.random.default_rng(seed)
    centers = np.eye(16)[:3] * 5
    y = np.repeat(np.arange(3), 50)
    return (centers[y] + rng.normal(0, 0.1, (150, 16))).astype(np.float32), y


def test_softmax_and_energy_rank_a_flat_logit_highest():
    logit = np.array([[10.0, 0, 0], [0, 9, 0], [0.1, 0, 0.1]])
    known = np.arange(3)
    assert sc.softmax_score(logit, known).argmax() == 2
    assert sc.energy_score(logit, known).argmax() == 2


def test_scores_ignore_the_held_out_class():
    logit = np.array([[0.0, 0, 20], [5, 0, 0]])  # row 0 is sure only of class 2
    known = np.array([0, 1])
    assert sc.softmax_score(logit, known).argmax() == 0
    assert sc.energy_score(logit, known).argmax() == 0


def test_embedding_scores_rank_an_outlier_highest():
    x, y = clusters()
    outlier = np.full((1, 16), 3.0, dtype=np.float32)
    queries = np.concatenate([x[[0, 60, 120]], outlier])
    means, precision = sc.fit_mahalanobis(x, y, np.arange(3))
    assert sc.mahalanobis_score(queries, means, precision).argmax() == 3
    assert knn_distance(queries, x, 5).argmax() == 3


def test_mahalanobis_is_zero_at_a_class_mean():
    x, y = clusters()
    means, precision = sc.fit_mahalanobis(x, y, np.arange(3))
    assert sc.mahalanobis_score(means, means, precision) == pytest.approx(0, abs=1e-6)


def test_all_scores_returns_every_score_per_query():
    x, y = clusters()
    head = (np.random.default_rng(1).normal(size=(4, 16)), np.zeros(4))
    out = sc.all_scores((x, y), x[:7], head, known=np.arange(3))
    assert set(out) == set(sc.SCORES)
    assert all(v.shape == (7,) for v in out.values())


def test_threshold_flags_the_chosen_share():
    known = np.arange(1000, dtype=float)
    assert (known > threshold(known, 0.05)).mean() == pytest.approx(0.05, abs=0.002)
