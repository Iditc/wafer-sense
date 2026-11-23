"""Unit tests: embedding tests on clusters with a known answer."""

import numpy as np
import pytest

from src.data.plot_utils import CLASSES
from src.models import embed_eval as ee
from src.models.metrics import OK_INDEX

CENTER, SCRATCH = CLASSES.index("Center"), CLASSES.index("Scratch")


def cluster(axis: int, n: int, seed: int) -> np.ndarray:
    """n vectors close to one coordinate axis."""
    rng = np.random.default_rng(seed)
    x = 0.02 * rng.standard_normal((n, 4)).astype(np.float32)
    x[:, axis] += 1
    return x


def split_of(counts: dict[int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings and labels: each class is its own cluster."""
    axes = {OK_INDEX: 0, CENTER: 1, SCRATCH: 2}
    x = np.concatenate([cluster(axes[c], n, seed + c) for c, n in counts.items()])
    y = np.concatenate([np.full(n, c) for c, n in counts.items()])
    return x, y


@pytest.fixture
def train() -> tuple[np.ndarray, np.ndarray]:
    return split_of({OK_INDEX: 30, CENTER: 20, SCRATCH: 20}, seed=0)


@pytest.fixture
def val() -> tuple[np.ndarray, np.ndarray]:
    return split_of({OK_INDEX: 20, CENTER: 10, SCRATCH: 10}, seed=100)


def test_nearest_finds_the_row_itself_first():
    x = cluster(0, 5, 0)
    sims, idx = ee.nearest(x, x, k=1)
    assert idx[:, 0].tolist() == list(range(5))
    assert sims[:, 0] == pytest.approx(1.0, abs=1e-5)


def test_nearest_sorts_by_similarity():
    gallery = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    sims, idx = ee.nearest(np.array([[1.0, 0.0]], dtype=np.float32), gallery, k=3)
    assert idx[0].tolist() == [0, 1, 2]
    assert sims[0] == pytest.approx([1.0, 0.8, 0.0], abs=1e-5)


def test_nearest_ignores_vector_length():
    gallery = np.array([[10.0, 0.0], [0.0, 0.1]], dtype=np.float32)
    _, idx = ee.nearest(np.array([[0.0, 1.0]], dtype=np.float32), gallery, k=1)
    assert idx[0, 0] == 1


def test_nearest_result_does_not_depend_on_chunk_size():
    query, gallery = cluster(1, 9, 1), cluster(1, 20, 2)
    assert np.array_equal(
        ee.nearest(query, gallery, 3, chunk=2)[1],
        ee.nearest(query, gallery, 3, chunk=256)[1],
    )


def test_knn_distance_is_zero_for_identical_and_one_for_orthogonal():
    gallery = np.array([[1.0, 0.0]], dtype=np.float32)
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert ee.knn_distance(query, gallery, 1) == pytest.approx([0.0, 1.0], abs=1e-5)


def test_similar_cases_is_perfect_on_separate_clusters(train, val):
    result = ee.similar_cases(train, val, k=5)
    assert result["macro"] == pytest.approx(1.0)
    assert set(result["per_class"]) == {"Center", "Scratch"}


def test_similar_cases_ignores_ok_queries(train, val):
    result = ee.similar_cases(train, val, k=5)
    assert "O.K." not in result["per_class"]


def test_similar_cases_detects_wrong_neighbors(val):
    x, y = split_of({OK_INDEX: 30, CENTER: 20, SCRATCH: 20}, seed=0)
    swapped = np.where(y == CENTER, SCRATCH, np.where(y == SCRATCH, CENTER, y))
    assert ee.similar_cases((x, swapped), val, k=5)["macro"] == pytest.approx(0.0)


def test_anomaly_scores_defects_above_ok(train, val):
    score = ee.anomaly(train, val, k=5)
    defect = val[1] != OK_INDEX
    assert score[defect].min() > score[~defect].max()


def test_new_type_stands_out_when_absent_from_gallery(train, val):
    result = ee.new_type(train, val, "Scratch", k=5)
    assert result["auroc_vs_known_defects"] == 1.0
    assert result["auroc_vs_all_known"] == 1.0
    assert result["flagged_as_new"] == 1.0


def test_new_type_does_not_use_held_out_wafers_in_the_gallery(val):
    x, y = split_of({OK_INDEX: 30, CENTER: 20, SCRATCH: 20}, seed=0)
    without = (x[y != SCRATCH], y[y != SCRATCH])
    assert ee.new_type((x, y), val, "Scratch") == ee.new_type(without, val, "Scratch")


def test_probe_learns_clusters_and_reports_train_size(train, val):
    result = ee.probe(train, val, frac=0.5)
    assert result["accuracy"] == 1.0
    assert result["n_train"] == 35
