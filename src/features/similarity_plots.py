"""Figures for the day 3b similarity comparison."""

import matplotlib.pyplot as plt
import numpy as np

from src.data.plot_utils import CLASSES, display, draw_map, save, setup_style
from src.features.retrieval import Gallery
from src.features.retrieval_review import split_maps

MATCH, MISMATCH = "#1b7f3b", "#c0392b"
LABELS = {
    "cosine": "Cosine (current)",
    "euclidean": "Euclidean",
    "whitened": "Whitened cosine",
    "rot_mean": "Mean of 8 rotations",
    "block3": "Conv block 3",
    "overlap": "Die overlap (no model)",
}


def plot_comparison(results: dict) -> None:
    """Methods x defect types: precision@5, with macro and die-overlap columns."""
    setup_style()
    methods = list(results)
    types = list(results[methods[0]]["per_class"])
    columns = types + ["Macro P@5", "Die overlap"]
    values = np.array(
        [
            [results[m]["per_class"][t] for t in types]
            + [results[m]["precision_at_5"], results[m]["overlap"]]
            for m in methods
        ]
    )
    fig, ax = plt.subplots(figsize=(22, 9))
    ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(methods)):
        for j in range(len(columns)):
            color = "white" if values[i, j] > 0.6 else "black"
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=color)
    ax.axvline(len(types) - 0.5, color="black", linewidth=3)
    ax.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
    ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods])
    ax.set_title("Similar-case search by method (val, k=5, one neighbor per lot)")
    save(fig, "similarity_comparison.png")


def plot_method_neighbors(
    pos: int, found: dict, gallery: Gallery, filename: str
) -> None:
    """One val query (top left), then its 5 neighbors under every method, one row each.

    pos: position in found["queries"]; found holds "rows" per method, "queries"
    (val positions) and "labels" (their classes), as returned by similarity_compare.run.
    """
    setup_style()
    maps, val_rows = split_maps("val")
    _, train_rows = split_maps("train")
    methods = list(found["rows"])
    k = found["rows"][methods[0]].shape[1]
    query_class = found["labels"][pos]
    fig, axes = plt.subplots(
        len(methods), k + 1, figsize=(3.8 * (k + 1), 4.2 * len(methods))
    )
    query_map = maps[val_rows[found["queries"][pos]]]
    for i, method in enumerate(methods):
        draw_map(axes[i, 0], query_map, LABELS[method])
        axes[i, 0].title.set_fontweight("bold")
        for j, g in enumerate(found["rows"][method][pos]):
            label = gallery.labels[g]
            draw_map(axes[i, j + 1], maps[train_rows[g]], display(CLASSES[label]))
            axes[i, j + 1].title.set_color(MATCH if label == query_class else MISMATCH)
    fig.suptitle(
        f"{display(CLASSES[query_class])} query (left, titled by method) and its 5 neighbors"
    )
    fig.tight_layout()
    save(fig, filename)
