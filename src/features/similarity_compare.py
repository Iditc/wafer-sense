"""Day 3b: compare similarity methods for the similar-case search (val only).

Every method answers the same 3,821 val defect queries from the same gallery of
labeled train wafers, with k=5 and at most one neighbor per lot. Each is scored by
precision@5 per defect type and by the label-free defect-die overlap. Gaps to
cosine come with a paired bootstrap 95% interval over queries.
Run: python -m src.features.similarity_compare  (needs src.features.view_embeddings)
"""

import json
import time

import numpy as np

from src.config import METRICS_DIR
from src.data.plot_utils import CLASSES, display
from src.features import similarity as sim
from src.features.frozen import load_frozen_embeddings
from src.features.retrieval import K, MAX_PER_LOT, frozen_gallery
from src.features.retrieval_review import pick_queries, split_maps
from src.features.similarity_plots import plot_comparison, plot_method_neighbors
from src.features.view_embeddings import VIEWS_FILE
from src.models.metrics import OK_INDEX

METHODS = ("cosine", "euclidean", "whitened", "rot_mean", "block3", "overlap")
EXAMPLE_TYPES = ("Scratch", "Loc")
CANDIDATES = 200  # nearest rows fetched before the per-lot limit
BOOTSTRAP = 1000
SEED = 0


def load_vectors() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(train, val) vectors of every embedding method."""
    train, _ = load_frozen_embeddings("train")
    val, _ = load_frozen_embeddings("val")
    mean, matrix = sim.fit_whitening(train)
    views = np.load(VIEWS_FILE)
    return {
        "cosine": (train, val),
        "euclidean": (train, val),
        "whitened": (sim.whiten(train, mean, matrix), sim.whiten(val, mean, matrix)),
        "rot_mean": (views["train_rot_mean"], views["val_rot_mean"]),
        "block3": (views["train_block3"], views["val_block3"]),
    }


def defect_maps(split: str) -> np.ndarray:
    """Pooled 32x32 defect maps of a split's labeled wafers, in embedding order."""
    maps, rows = split_maps(split)
    return np.concatenate(
        [
            sim.pooled_defects(maps[rows[i : i + 4096]])
            for i in range(0, len(rows), 4096)
        ]
    )


def candidates(method: str, vectors: dict, q: np.ndarray, maps: dict) -> np.ndarray:
    """Candidate gallery rows of every defect query for one method."""
    if method == "overlap":
        return sim.overlap_candidates(maps["val"][q], maps["train"], CANDIDATES)[1]
    train, val = vectors[method]
    metric = "euclidean" if method == "euclidean" else "cosine"
    return sim.vector_candidates(val[q], train, CANDIDATES, metric)[1]


def macro(hits: np.ndarray, labels: np.ndarray) -> float:
    """Mean over defect types of the per-query scores."""
    return float(np.mean([hits[labels == c].mean() for c in np.unique(labels)]))


def bootstrap_gap(
    a: np.ndarray, b: np.ndarray, labels: np.ndarray, seed: int = SEED
) -> list[float]:
    """Paired 95% interval of macro(a) - macro(b), resampling queries."""
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, len(labels), len(labels))
        gaps.append(macro(a[pick], labels[pick]) - macro(b[pick], labels[pick]))
    return [round(float(v), 4) for v in np.percentile(gaps, [2.5, 97.5])]


def score(rows: np.ndarray, gallery, labels: np.ndarray, q_maps, g_maps) -> dict:
    """Per-query precision@k and defect-die overlap for one method's neighbors."""
    return {
        "precision": (gallery.labels[rows] == labels[:, None]).mean(axis=1),
        "overlap": sim.neighbor_overlap(q_maps, g_maps[rows]).mean(axis=1),
    }


def summary(scores: dict, labels: np.ndarray, base: dict, seconds: float) -> dict:
    """JSON-ready result of one method, with its gaps to cosine."""
    per_class = {
        display(CLASSES[c]): round(float(scores["precision"][labels == c].mean()), 4)
        for c in np.unique(labels)
    }
    return {
        "precision_at_5": round(macro(scores["precision"], labels), 4),
        "overlap": round(macro(scores["overlap"], labels), 4),
        "precision_gap_ci": bootstrap_gap(
            scores["precision"], base["precision"], labels
        ),
        "overlap_gap_ci": bootstrap_gap(scores["overlap"], base["overlap"], labels),
        "ms_per_query": round(seconds * 1000 / len(labels), 3),
        "per_class": per_class,
    }


def run() -> tuple[dict, dict]:
    """Neighbors and scores of every method; returns the results and the neighbor rows."""
    gallery = frozen_gallery()
    _, y = load_frozen_embeddings("val")
    q = np.flatnonzero(y != OK_INDEX)
    labels = y[q]
    vectors = load_vectors()
    maps = {split: defect_maps(split) for split in ("train", "val")}

    all_rows, all_scores, results = {}, {}, {}
    for method in METHODS:
        start = time.time()
        cand = candidates(method, vectors, q, maps)
        seconds = time.time() - start
        rows = sim.keep_per_lot(cand, gallery.lots, K, MAX_PER_LOT)
        all_rows[method] = rows
        all_scores[method] = score(rows, gallery, labels, maps["val"][q], maps["train"])
        results[method] = summary(
            all_scores[method], labels, all_scores["cosine"], seconds
        )
        print(method, {k: v for k, v in results[method].items() if k != "per_class"})
    return results, {"rows": all_rows, "queries": q, "labels": labels}


def main() -> None:
    """Run the comparison, save the metrics and the figures."""
    results, found = run()
    out = {"split": "val", "k": K, "max_per_lot": MAX_PER_LOT, "methods": results}
    (METRICS_DIR / "similarity_val.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    plot_comparison(results)
    gallery = frozen_gallery()
    for name in EXAMPLE_TYPES:
        picks = pick_queries(
            found["labels"], (name,), np.ones(len(found["labels"]), bool)
        )
        plot_method_neighbors(
            picks[0], found, gallery, f"similarity_examples_{name.lower()}.png"
        )


if __name__ == "__main__":
    main()
