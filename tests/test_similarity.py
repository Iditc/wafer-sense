"""Tests for the day 3b similarity methods and rotation-averaged embeddings."""

import numpy as np
import pytest
import torch

from src.features import similarity as sim
from src.features.view_embeddings import symmetries, view_embeddings
from src.models.cnn import WaferCNN


def random_maps(n: int = 6, size: int = 64, seed: int = 0) -> np.ndarray:
    """Wafer-like 0/1/2 maps with a random block of defects each."""
    rng = np.random.default_rng(seed)
    maps = np.ones((n, size, size), dtype=np.uint8)
    for m in maps:
        r, c = rng.integers(0, size - 12, 2)
        m[r : r + rng.integers(4, 12), c : c + rng.integers(2, 12)] = 2
    return maps


def test_pooled_defects_marks_a_cell_with_any_defect():
    maps = np.ones((1, 4, 4), dtype=np.uint8)
    maps[0, 0, 1] = 2
    assert sim.pooled_defects(maps, pool=2).tolist() == [[1, 0, 0, 0]]


def test_defect_symmetries_are_eight_distinct_views():
    flat = sim.pooled_defects(random_maps(1))
    views = sim.defect_symmetries(flat)
    assert views.shape == (8, 1, flat.shape[1])
    assert len({v.tobytes() for v in views[:, 0]}) == 8


def test_overlap_is_one_for_a_rotated_copy():
    flat = sim.pooled_defects(random_maps(1))
    rotated = sim.defect_symmetries(flat)[5]  # rotation plus flip
    score = sim.neighbor_overlap(flat, rotated[:, None, :])
    assert score[0, 0] == pytest.approx(1.0)


def test_iou_is_zero_when_both_maps_are_empty():
    assert sim.iou(np.zeros(1), np.zeros(1), np.zeros(1))[0] == 0


def test_overlap_search_finds_the_identical_wafer_first():
    gallery = sim.pooled_defects(random_maps(20))
    sims, rows = sim.overlap_candidates(gallery[[3, 7]], gallery, n=4, chunk=1)
    assert rows[:, 0].tolist() == [3, 7]
    assert np.allclose(sims[:, 0], 1.0)
    assert (np.diff(sims, axis=1) <= 0).all()


def test_overlap_search_finds_a_rotated_copy():
    gallery = sim.pooled_defects(random_maps(20))
    query = sim.defect_symmetries(gallery[[4]])[2]  # 180 degree turn
    _, rows = sim.overlap_candidates(query, gallery, n=1)
    assert rows[0, 0] == 4


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_vector_search_finds_the_identical_row_first(metric):
    x = np.random.default_rng(0).standard_normal((50, 16)).astype(np.float32)
    _, rows = sim.vector_candidates(x[[5, 9]], x, n=3, metric=metric)
    assert rows[:, 0].tolist() == [5, 9]


def test_vector_search_rejects_an_unknown_metric():
    with pytest.raises(ValueError):
        sim.vector_candidates(np.ones((1, 2)), np.ones((3, 2)), n=1, metric="manhattan")


def test_whitening_gives_unit_variance_directions():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5000, 4)) * np.array([10, 8, 6, 5])
    mean, matrix = sim.fit_whitening(x)
    cov = np.cov(sim.whiten(x, mean, matrix), rowvar=False)
    assert np.allclose(np.diag(cov), 1, atol=0.05)  # shrink keeps it just under 1
    assert np.abs(cov - np.diag(np.diag(cov))).max() < 0.05


def test_whitening_damps_near_empty_directions():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5000, 3)) * np.array([10, 10, 0.01])
    mean, matrix = sim.fit_whitening(x)
    variance = np.diag(np.cov(sim.whiten(x, mean, matrix), rowvar=False))
    assert variance.min() < 0.01  # noise is not blown up to unit variance


def test_keep_per_lot_takes_first_rows_of_new_lots():
    lots = np.array(["a", "a", "b", "c", "b"])
    rows = sim.keep_per_lot(np.array([[0, 1, 2, 4, 3]]), lots, k=3)
    assert rows.tolist() == [[0, 2, 3]]


def test_keep_per_lot_fails_when_lots_run_out():
    with pytest.raises(ValueError, match="lots"):
        sim.keep_per_lot(np.array([[0, 1]]), np.array(["a", "a"]), k=2)


def test_symmetries_of_a_batch():
    x = torch.arange(2 * 2 * 3 * 3, dtype=torch.float32).reshape(2, 2, 3, 3)
    views = symmetries(x)
    assert len(views) == 8 and torch.equal(views[0], x)
    assert len({v.numpy().tobytes() for v in views}) == 8


def test_rotation_mean_is_the_same_for_every_rotation():
    torch.manual_seed(0)
    model = WaferCNN().eval()
    x = (torch.rand(3, 2, 64, 64) > 0.5).float()
    _, base, _ = view_embeddings(model, x)
    for view in symmetries(x)[1:]:
        _, rotated, _ = view_embeddings(model, view)
        assert torch.allclose(base, rotated, atol=1e-5)


def test_view_embeddings_shapes_and_plain_matches_embed():
    model = WaferCNN().eval()
    x = torch.rand(2, 2, 64, 64)
    plain, rot_mean, block3 = view_embeddings(model, x)
    assert plain.shape == rot_mean.shape == (2, 256) and block3.shape == (2, 128)
    assert torch.allclose(plain, model.embed(x))


def test_macro_weights_every_type_equally():
    from src.features.similarity_compare import macro

    hits = np.array([1.0, 1.0, 1.0, 0.0])
    assert macro(hits, np.array([1, 1, 1, 2])) == pytest.approx(0.5)


def test_bootstrap_gap_is_zero_for_identical_methods_and_sees_a_real_gap():
    from src.features.similarity_compare import bootstrap_gap

    labels = np.repeat([1, 2], 50)
    a = np.random.default_rng(0).random(100)
    assert bootstrap_gap(a, a, labels) == [0.0, 0.0]
    low, high = bootstrap_gap(a + 0.2, a, labels)
    assert low == pytest.approx(0.2) and high == pytest.approx(0.2)
