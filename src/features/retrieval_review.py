"""Day 3 review of similar-case search on val (test stays closed).

Measures precision at several k, how often neighbors come from the same lot, what
the weak defect types are confused with, and draws query + neighbor examples.
Run: python -m src.features.retrieval_review
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from src.config import METRICS_DIR
from src.data.cache_maps import cache_paths
from src.data.plot_utils import CLASSES, display, draw_map, save, setup_style
from src.features.frozen import load_frozen_embeddings
from src.features.retrieval import (
    Gallery,
    frozen_gallery,
    lot_names,
    precision_at_k,
    split_wafer_ids,
)
from src.models.dataset import split_index
from src.models.metrics import OK_INDEX, plot_confusion

KS = (1, 5, 10, 20)
LOT_LIMITS = (1, 2)
EXAMPLE_TYPES = ("Loc", "Scratch", "Edge-Loc", "Donut")
FAILURE_TYPES = ("Loc", "Scratch")
SEED = 0
MATCH, MISMATCH = "#1b7f3b", "#c0392b"


def lot_stats(lots: np.ndarray) -> dict:
    """How often the k neighbors of a query share a lot (lots: one row per query)."""
    distinct = np.array([len(set(row)) for row in lots])
    k = lots.shape[1]
    return {
        "k": k,
        "mean_distinct_lots": float(distinct.mean()),
        "share_with_a_shared_lot": float((distinct < k).mean()),
        "share_all_from_one_lot": float((distinct == 1).mean()),
    }


def neighbor_label_counts(
    neighbor_labels: np.ndarray, query_labels: np.ndarray
) -> np.ndarray:
    """Counts of neighbor classes per query class (rows: query, columns: neighbor)."""
    counts = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for query, labels in zip(query_labels, neighbor_labels):
        np.add.at(counts[query], labels, 1)
    return counts


def pick_queries(
    labels: np.ndarray, types: tuple[str, ...], allowed: np.ndarray, seed: int = SEED
) -> list[int]:
    """One random query per type among the allowed positions."""
    rng = np.random.default_rng(seed)
    picks = []
    for name in types:
        pool = np.flatnonzero((labels == CLASSES.index(name)) & allowed)
        if len(pool):
            picks.append(int(rng.choice(pool)))
    return picks


def split_maps(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Memory-mapped 64x64 maps and the cache rows of a split's labeled wafers."""
    rows, _ = split_index(split)
    maps_path, _ = cache_paths(64, "labeled")
    return np.load(maps_path, mmap_mode="r"), rows


def plot_neighbors(
    queries: list[int], val: dict, gallery: Gallery, title: str, filename: str
) -> None:
    """One row per query: the query, then its neighbors with class, similarity and lot."""
    setup_style()
    maps, val_rows = split_maps("val")
    _, train_rows = split_maps("train")
    k = val["rows"].shape[1]
    fig, axes = plt.subplots(
        len(queries), k + 1, figsize=(3.8 * (k + 1), 5.4 * len(queries))
    )
    for i, q in enumerate(queries):
        query_class = val["y"][q]
        draw_map(
            axes[i, 0],
            maps[val_rows[q]],
            f"QUERY\n{display(CLASSES[query_class])}\n{val['lots'][q]}",
        )
        for j in range(k):
            g = val["rows"][q, j]
            label = gallery.labels[g]
            draw_map(
                axes[i, j + 1],
                maps[train_rows[g]],
                f"{display(CLASSES[label])}\nsim {val['sims'][q, j]:.2f}\n{gallery.lots[g]}",
            )
            axes[i, j + 1].title.set_color(MATCH if label == query_class else MISMATCH)
    fig.suptitle(title)
    fig.tight_layout(h_pad=3.0)
    save(fig, filename)


def plot_precision_at_k(results: dict[int, dict]) -> None:
    """Precision at k for every defect type, with the macro average."""
    setup_style()
    ks = sorted(results)
    fig, ax = plt.subplots(figsize=(13, 8))
    for name in results[ks[0]]["per_class"]:
        ax.plot(ks, [results[k]["per_class"][name] for k in ks], marker="o", label=name)
    ax.plot(
        ks,
        [results[k]["macro"] for k in ks],
        marker="o",
        color="black",
        linewidth=4,
        label="Macro average",
    )
    ax.set(
        xlabel="k (neighbors returned)",
        ylabel="Share with the same defect type",
        xticks=ks,
        ylim=(0.5, 1.0),
        title="Similar-case precision at k (val)",
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, "retrieval_precision_at_k.png")


def main() -> None:
    """Run every review measurement, save metrics and figures."""
    gallery = frozen_gallery()
    x, y = load_frozen_embeddings("val")
    defect = y != OK_INDEX
    sims, rows = gallery.search(x, 5)
    val = {
        "y": y,
        "lots": lot_names(split_wafer_ids("val")),
        "sims": sims,
        "rows": rows,
    }
    hits = (gallery.labels[rows] == y[:, None]).sum(axis=1)

    precision = {k: precision_at_k(gallery, x, y, k) for k in KS}
    limited = {m: precision_at_k(gallery, x, y, 5, max_per_lot=m) for m in LOT_LIMITS}
    limited_sims, limited_rows = gallery.search_per_lot(x[defect], 5, max_per_lot=1)
    counts = neighbor_label_counts(gallery.labels[rows[defect]], y[defect])
    # The example figures show what the search returns by default: one neighbor per lot.
    val["sims"][defect], val["rows"][defect] = limited_sims, limited_rows
    limited_hits = np.zeros(len(y), dtype=int)
    limited_hits[defect] = (gallery.labels[limited_rows] == y[defect][:, None]).sum(1)

    out = {
        "split": "val",
        "queries": int(defect.sum()),
        "precision_at_k": precision,
        "precision_at_5_max_per_lot": limited,
        "lot_sharing_top5": lot_stats(gallery.lots[rows[defect]]),
        "lot_sharing_top5_max_1_per_lot": lot_stats(gallery.lots[limited_rows]),
        "zero_hits_top5_by_class": {
            display(CLASSES[c]): int(((hits == 0) & (y == c)).sum())
            for c in np.unique(y[defect])
        },
        "neighbor_label_counts": counts.tolist(),
    }
    (METRICS_DIR / "retrieval_review_val.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )

    plot_precision_at_k(precision)
    plot_confusion(
        counts,
        "Classes of the 5 nearest train wafers (val defect queries)",
        "retrieval_neighbor_classes.png",
        xlabel="Neighbor class",
        ylabel="Query class",
    )
    plot_neighbors(
        pick_queries(y, EXAMPLE_TYPES, defect),
        val,
        gallery,
        "Similar cases: one random val query per type (one neighbor per lot)",
        "retrieval_examples.png",
    )
    plot_neighbors(
        pick_queries(y, FAILURE_TYPES, defect & (limited_hits == 0)),
        val,
        gallery,
        "Failures: no neighbor with the query's type (one neighbor per lot)",
        "retrieval_failures.png",
    )
    print(
        json.dumps(
            {k: v for k, v in out.items() if k != "neighbor_label_counts"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
