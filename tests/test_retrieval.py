"""Tests for similar-case search: exact results, train-only gallery, readable output."""

import numpy as np
import pandas as pd
import pytest

from src.config import FROZEN_MANIFEST, SPLIT_PARQUET
from src.data.plot_utils import CLASSES
from src.features import retrieval
from src.features.retrieval import Gallery, precision_at_k
from src.models import embed_eval
from src.models.metrics import OK_INDEX

CENTER, SCRATCH = CLASSES.index("Center"), CLASSES.index("Scratch")


def random_gallery(
    n: int = 200, dim: int = 16, seed: int = 0
) -> tuple[Gallery, np.ndarray]:
    """Gallery of random vectors, labeled round-robin, with ids 1000, 1001, ..."""
    x = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return Gallery.build(x, np.arange(n) % 9, np.arange(1000, 1000 + n)), x


def clusters(counts: dict[int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings where each class is its own tight cluster."""
    rng = np.random.default_rng(seed)
    axis = {OK_INDEX: 0, CENTER: 1, SCRATCH: 2}
    parts = []
    for c, n in counts.items():
        x = 0.02 * rng.standard_normal((n, 4)).astype(np.float32)
        x[:, axis[c]] += 1
        parts.append(x)
    labels = np.concatenate([np.full(n, c) for c, n in counts.items()])
    return np.concatenate(parts), labels


def test_faiss_matches_brute_force_search():
    gallery, x = random_gallery()
    queries = np.random.default_rng(1).standard_normal((30, 16)).astype(np.float32)
    sims, rows = gallery.search(queries, 5)
    brute_sims, brute_rows = embed_eval.nearest(queries, x, 5)
    assert np.array_equal(rows, brute_rows)
    assert np.allclose(sims, brute_sims, atol=1e-5)


def test_a_wafer_finds_itself_first():
    gallery, x = random_gallery()
    sims, rows = gallery.search(x, 1)
    assert rows[:, 0].tolist() == list(range(len(x)))
    assert np.allclose(sims[:, 0], 1.0, atol=1e-5)


def test_results_are_sorted_by_similarity():
    gallery, _ = random_gallery()
    sims, _ = gallery.search(np.ones(16), 10)
    assert (np.diff(sims[0]) <= 1e-7).all()


def test_search_ignores_vector_length():
    gallery, x = random_gallery()
    assert np.array_equal(gallery.search(x[:5], 3)[1], gallery.search(7 * x[:5], 3)[1])


def test_search_does_not_change_the_query():
    gallery, x = random_gallery()
    query = x[:3].copy()
    gallery.search(query, 2)
    assert np.array_equal(query, x[:3])


def test_neighbors_are_readable_gallery_records():
    gallery, x = random_gallery()
    records = gallery.neighbors(x[CENTER], k=3)
    assert len(records) == 3
    assert records[0] == {
        "wafer_id": 1000 + CENTER,
        "label": "Center",
        "similarity": 1.0,
    }
    assert all(1000 <= r["wafer_id"] < 1200 for r in records)


@pytest.mark.parametrize("k", [0, 201])
def test_k_must_fit_the_gallery(k):
    gallery, x = random_gallery()
    with pytest.raises(ValueError, match="k must be"):
        gallery.search(x[:1], k)


def test_build_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        Gallery.build(np.ones((3, 4)), np.zeros(3), np.arange(2))


def test_precision_matches_the_embedding_test():
    train = clusters({OK_INDEX: 30, CENTER: 20, SCRATCH: 20}, seed=0)
    val = clusters({OK_INDEX: 10, CENTER: 8, SCRATCH: 8}, seed=1)
    gallery = Gallery.build(*train, np.arange(len(train[1])))
    result = precision_at_k(gallery, *val)
    assert result["macro"] == pytest.approx(
        embed_eval.similar_cases(train, val)["macro"]
    )
    assert result["macro"] == 1.0 and result["hit_rate"] == 1.0


def test_precision_skips_ok_queries_and_counts_misses():
    train = clusters({OK_INDEX: 30, CENTER: 20}, seed=0)
    gallery = Gallery.build(train[0], train[1], np.arange(50))
    queries, labels = clusters({OK_INDEX: 5, CENTER: 5}, seed=1)
    labels = np.where(labels == CENTER, SCRATCH, labels)  # no Scratch in the gallery
    result = precision_at_k(gallery, queries, labels)
    assert result["per_class"] == {"Scratch": 0.0} and result["hit_rate"] == 0.0


def test_similarity_gap_is_tiny_for_exact_search():
    gallery, x = random_gallery()
    assert retrieval.max_similarity_gap(gallery, x, x[:20]) < 1e-5


@pytest.mark.data
def test_real_gallery_holds_only_train_wafers():
    if not FROZEN_MANIFEST.exists() or not SPLIT_PARQUET.exists():
        pytest.skip("frozen model or split not found")
    gallery = retrieval.frozen_gallery()
    split = pd.read_parquet(SPLIT_PARQUET, columns=["wafer_id", "split"]).set_index(
        "wafer_id"
    )
    assert len(np.unique(gallery.wafer_ids)) == len(gallery)
    assert (split.loc[gallery.wafer_ids, "split"] == "train").all()

    val_x, val_y = retrieval.load_frozen_embeddings("val")
    _, rows = gallery.search(val_x[:500], 5)
    assert (split.loc[gallery.wafer_ids[rows.ravel()], "split"] == "train").all()
    assert precision_at_k(gallery, val_x, val_y)["macro"] == pytest.approx(
        0.866, abs=0.001
    )
