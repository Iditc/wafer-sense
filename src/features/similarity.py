"""Similarity methods compared on day 3b, and a label-free way to score them.

Embedding methods (frozen model):
  cosine    - angle between embeddings (the current search)
  euclidean - straight-line distance, so embedding length counts too
  whitened  - cosine after PCA whitening on train: every direction gets equal
              variance, so a few dominant directions no longer decide alone
  rot_mean  - cosine on the embedding averaged over the 8 rotations and flips
  block3    - cosine on the pooled third conv block (earlier, more visual)
No-model method:
  overlap   - intersection over union of the defective dies, best of the 8
              rotations and flips, on 32x32 maps (a die counts if any of its
              2x2 cells is defective)
Label-free score of any method: the mean defect-die overlap between a query
and its neighbors. The overlap method optimizes this score directly, so for it
the score is an upper reference, not a fair result.
"""

import faiss
import numpy as np

from src.features.retrieval import limit_per_lot, normalize

POOL = 2  # 64x64 -> 32x32 for die overlap
WHITEN_SHRINK = 1e-2  # added to every eigenvalue, as a share of the mean eigenvalue
CHUNK = 256


def pooled_defects(maps: np.ndarray, pool: int = POOL) -> np.ndarray:
    """Flat 0/1 defect maps, a cell set if any die in its pool x pool block is defective."""
    maps = np.asarray(maps)
    n, h, w = maps.shape
    blocks = (maps == 2).reshape(n, h // pool, pool, w // pool, pool)
    return blocks.any(axis=(2, 4)).reshape(n, -1).astype(np.float32)


def defect_symmetries(flat: np.ndarray) -> np.ndarray:
    """The 8 rotations and flips of flat square maps (8 x N x D)."""
    side = int(round(np.sqrt(flat.shape[1])))
    grid = flat.reshape(-1, side, side)
    turns = [np.rot90(grid, k, axes=(1, 2)) for k in range(4)]
    views = turns + [t[:, :, ::-1] for t in turns]
    return np.stack([v.reshape(len(flat), -1) for v in views])


def iou(inter: np.ndarray, size_a: np.ndarray, size_b: np.ndarray) -> np.ndarray:
    """Intersection over union from overlap counts and set sizes; 0 when both are empty."""
    union = size_a + size_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def overlap_candidates(
    queries: np.ndarray, gallery: np.ndarray, n: int, chunk: int = CHUNK
) -> tuple[np.ndarray, np.ndarray]:
    """Best defect-die overlap (over 8 symmetries) and gallery rows of the n best matches."""
    g_size = gallery.sum(axis=1)[None, :]
    sims = np.empty((len(queries), n), dtype=np.float32)
    rows = np.empty((len(queries), n), dtype=np.int64)
    for start in range(0, len(queries), chunk):
        views = defect_symmetries(queries[start : start + chunk])
        q_size = views[0].sum(axis=1)[:, None]
        best = np.zeros((views.shape[1], len(gallery)), dtype=np.float32)
        for view in views:
            np.maximum(best, iou(view @ gallery.T, q_size, g_size), out=best)
        top = np.argpartition(-best, n - 1, axis=1)[:, :n]
        order = np.argsort(-np.take_along_axis(best, top, axis=1), axis=1)
        rows[start : start + chunk] = np.take_along_axis(top, order, axis=1)
        sims[start : start + chunk] = np.take_along_axis(
            best, rows[start : start + chunk], axis=1
        )
    return sims, rows


def fit_whitening(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and PCA whitening matrix learned from train embeddings only."""
    mean = train.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(train - mean, rowvar=False))
    shrink = WHITEN_SHRINK * values.mean()
    return mean, vectors / np.sqrt(np.clip(values, 0, None) + shrink)


def whiten(x: np.ndarray, mean: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Embeddings in the whitened space."""
    return ((x - mean) @ matrix).astype(np.float32)


def vector_candidates(
    queries: np.ndarray, gallery: np.ndarray, n: int, metric: str = "cosine"
) -> tuple[np.ndarray, np.ndarray]:
    """Similarity and gallery rows of the n nearest rows; Euclidean returns -distance."""
    if metric == "cosine":
        index = faiss.IndexFlatIP(gallery.shape[1])
        index.add(normalize(gallery))
        return index.search(normalize(queries), n)
    if metric == "euclidean":
        index = faiss.IndexFlatL2(gallery.shape[1])
        index.add(np.ascontiguousarray(gallery, dtype=np.float32))
        dist, rows = index.search(np.ascontiguousarray(queries, dtype=np.float32), n)
        return -np.sqrt(np.maximum(dist, 0)), rows
    raise ValueError(f"Unknown metric '{metric}'")


def keep_per_lot(
    candidates: np.ndarray, lots: np.ndarray, k: int, max_per_lot: int = 1
) -> np.ndarray:
    """The first k candidate rows of every query with at most max_per_lot per lot."""
    out = np.empty((len(candidates), k), dtype=np.int64)
    for i, row in enumerate(candidates):
        keep = limit_per_lot(row, lots, k, max_per_lot)
        if len(keep) < k:
            raise ValueError(f"Query {i}: only {len(keep)} lots among the candidates")
        out[i] = row[keep]
    return out


def neighbor_overlap(queries: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Best-symmetry defect-die overlap between each query and each of its neighbors.

    queries: N x D flat defect maps; neighbors: N x k x D. Returns N x k.
    """
    views = defect_symmetries(queries)
    inter = np.einsum("snd,nkd->snk", views, neighbors)
    q_size = queries.sum(axis=1)[None, :, None]
    n_size = neighbors.sum(axis=2)[None]
    return iou(inter, q_size, n_size).max(axis=0)
