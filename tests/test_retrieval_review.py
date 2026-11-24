"""Tests for the per-lot neighbor limit and the day 3 review measurements."""

import numpy as np
import pytest

from src.data.plot_utils import CLASSES
from src.features.retrieval import Gallery, limit_per_lot, precision_at_k
from src.features.retrieval_review import lot_stats, neighbor_label_counts, pick_queries

LOC, SCRATCH = CLASSES.index("Loc"), CLASSES.index("Scratch")


def lot_gallery(
    n_lots: int = 20, per_lot: int = 5, seed: int = 0
) -> tuple[Gallery, np.ndarray]:
    """Gallery where wafers of one lot are near-copies of each other."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_lots, 16)).astype(np.float32)
    x = np.repeat(centers, per_lot, axis=0)
    x += 0.001 * rng.standard_normal(x.shape).astype(np.float32)
    lots = np.repeat([f"lot{i}" for i in range(n_lots)], per_lot)
    labels = np.repeat(np.arange(n_lots) % 9, per_lot)
    return Gallery.build(x, labels, np.arange(len(x)), lots), x


def test_limit_keeps_order_and_skips_extra_wafers_of_a_lot():
    lots = np.array(["a", "a", "b", "a", "c", "b"])
    assert limit_per_lot(np.arange(6), lots, k=3, max_per_lot=1).tolist() == [0, 2, 4]
    assert limit_per_lot(np.arange(6), lots, k=4, max_per_lot=2).tolist() == [
        0,
        1,
        2,
        4,
    ]


def test_limit_returns_fewer_when_candidates_run_out():
    lots = np.array(["a", "a", "a"])
    assert limit_per_lot(np.arange(3), lots, k=3, max_per_lot=1).tolist() == [0]


def test_limit_must_be_positive():
    with pytest.raises(ValueError):
        limit_per_lot(np.arange(3), np.array(["a", "b", "c"]), k=2, max_per_lot=0)


@pytest.mark.parametrize("max_per_lot", [1, 2])
def test_search_per_lot_respects_limit_and_returns_k(max_per_lot):
    gallery, x = lot_gallery()
    _, rows = gallery.search_per_lot(x[:10], k=5, max_per_lot=max_per_lot)
    assert rows.shape == (10, 5)
    for row in rows:
        _, counts = np.unique(gallery.lots[row], return_counts=True)
        assert counts.max() <= max_per_lot
        assert len(set(row)) == 5


def test_search_per_lot_fetches_more_when_needed():
    gallery, x = lot_gallery(n_lots=20, per_lot=5)  # 5 results need about 25 candidates
    sims, rows = gallery.search_per_lot(x[0], k=12, max_per_lot=1)
    assert len(set(gallery.lots[rows[0]])) == 12
    assert (np.diff(sims[0]) <= 1e-6).all()


def test_search_per_lot_keeps_the_best_wafer_of_each_lot():
    gallery, x = lot_gallery()
    sims, rows = gallery.search_per_lot(x[0], k=1, max_per_lot=1)
    assert rows[0, 0] == 0 and sims[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_search_per_lot_fails_when_too_few_lots():
    gallery, x = lot_gallery(n_lots=3)
    with pytest.raises(ValueError, match="per-lot"):
        gallery.search_per_lot(x[0], k=4, max_per_lot=1)


def test_search_per_lot_needs_lots():
    x = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError, match="no lots"):
        Gallery.build(x, np.zeros(4), np.arange(4)).search_per_lot(x[0], k=2)


def test_neighbors_show_lots_and_use_the_limit_by_default():
    gallery, x = lot_gallery()
    records = gallery.neighbors(x[0], k=5)
    assert len({r["lot"] for r in records}) == 5
    unlimited = gallery.neighbors(x[0], k=5, max_per_lot=None)
    assert len({r["lot"] for r in unlimited}) == 1


def test_precision_with_limit_uses_distinct_lots():
    base, x = lot_gallery()
    # Only lot 0 is Scratch; every other lot is Loc.
    labels = np.where(base.lots == "lot0", SCRATCH, LOC)
    gallery = Gallery.build(x, labels, base.wafer_ids, base.lots)
    plain = precision_at_k(gallery, x[:5], labels[:5], k=5)
    limited = precision_at_k(gallery, x[:5], labels[:5], k=5, max_per_lot=1)
    assert plain["macro"] == pytest.approx(1.0)  # all 5 neighbors are its own lot
    assert limited["macro"] == pytest.approx(0.2)  # only one may come from its lot


def test_lot_stats():
    lots = np.array([["a", "b", "c"], ["a", "a", "b"], ["a", "a", "a"]])
    stats = lot_stats(lots)
    assert stats["mean_distinct_lots"] == pytest.approx(2.0)
    assert stats["share_with_a_shared_lot"] == pytest.approx(2 / 3)
    assert stats["share_all_from_one_lot"] == pytest.approx(1 / 3)


def test_neighbor_label_counts():
    counts = neighbor_label_counts(
        np.array([[LOC, LOC, 0], [SCRATCH, LOC, LOC]]), np.array([LOC, SCRATCH])
    )
    assert counts[LOC, LOC] == 2 and counts[LOC, 0] == 1
    assert counts[SCRATCH, SCRATCH] == 1 and counts[SCRATCH, LOC] == 2
    assert counts.sum() == 6


def test_pick_queries_respects_type_and_filter():
    labels = np.array([LOC, LOC, SCRATCH, SCRATCH, 0])
    allowed = np.array([False, True, True, False, True])
    assert pick_queries(labels, ("Loc", "Scratch", "Donut"), allowed) == [1, 2]
