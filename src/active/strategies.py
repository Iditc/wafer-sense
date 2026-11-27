"""Day 7: how the simulated engineer's next batch of wafers is chosen.

  random   - uniform over the unlabeled wafers (the baseline)
  entropy  - wafers whose predicted class probabilities are most spread out
  margin   - wafers where the top two classes are closest
  kcenter  - greedy k-center: each pick is the wafer farthest from every labeled
             or already-picked wafer in the embedding (covers unseen regions)
A strategy sees embeddings, the current model and which rows are labeled,
never the hidden labels.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

CHUNK = 4096  # rows per block when computing distances


@dataclass
class Round:
    """What a strategy may look at in one round."""

    x: np.ndarray
    model: object
    labeled: np.ndarray
    unlabeled: np.ndarray
    n: int
    rng: np.random.Generator


Strategy = Callable[[Round], np.ndarray]


def random_strategy(state: Round) -> np.ndarray:
    """n unlabeled rows drawn uniformly at random."""
    return state.rng.choice(state.unlabeled, size=state.n, replace=False)


def top_n(scores: np.ndarray, rows: np.ndarray, n: int) -> np.ndarray:
    """Rows with the n highest scores (ties broken by row order)."""
    return rows[np.argsort(-scores, kind="stable")[:n]]


def entropy_strategy(state: Round) -> np.ndarray:
    """n unlabeled rows with the highest entropy of the predicted probabilities."""
    p = state.model.predict_proba(state.x[state.unlabeled])
    entropy = -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1)
    return top_n(entropy, state.unlabeled, state.n)


def margin_strategy(state: Round) -> np.ndarray:
    """n unlabeled rows with the smallest gap between the two most likely classes."""
    p = np.sort(state.model.predict_proba(state.x[state.unlabeled]), axis=1)
    return top_n(-(p[:, -1] - p[:, -2]), state.unlabeled, state.n)


def unit(x: np.ndarray) -> np.ndarray:
    """Rows scaled to length 1, so Euclidean distance ranks like cosine distance."""
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def min_distance(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Distance from every point to its nearest center, computed in blocks."""
    out = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), CHUNK):
        block = points[start : start + CHUNK]
        best = np.full(len(block), np.inf, dtype=np.float32)
        for c in range(0, len(centers), CHUNK):
            # squared distance between unit rows: 2 - 2 * cosine similarity
            d = 2 - 2 * block @ centers[c : c + CHUNK].T
            best = np.minimum(best, d.min(axis=1))
        out[start : start + CHUNK] = best
    return np.sqrt(np.maximum(out, 0))


def kcenter_strategy(state: Round) -> np.ndarray:
    """Greedy k-center: repeatedly pick the unlabeled row farthest from all centers."""
    pool = unit(state.x[state.unlabeled]).astype(np.float32)
    dist = min_distance(pool, unit(state.x[state.labeled]).astype(np.float32))
    picks = []
    for _ in range(state.n):
        i = int(np.argmax(dist))
        picks.append(i)
        dist = np.minimum(dist, np.linalg.norm(pool - pool[i], axis=1))
        dist[i] = -1  # never pick the same row again
    return state.unlabeled[np.array(picks)]


STRATEGIES: dict[str, Strategy] = {
    "random": random_strategy,
    "entropy": entropy_strategy,
    "margin": margin_strategy,
    "kcenter": kcenter_strategy,
}
