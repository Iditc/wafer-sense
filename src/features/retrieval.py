"""Similar-case search: the k labeled train wafers closest to a query wafer.

Embeddings come from the frozen model. Similarity is cosine: vectors are scaled to
length 1 and searched by inner product in an exact FAISS index. The gallery holds
only labeled train wafers, so a search can never return a val or test wafer.
Run: python -m src.features.retrieval  (checks precision@5 against the embedding test)
"""

import argparse
import json
import time
from dataclasses import dataclass

import faiss
import numpy as np

from src.config import FROZEN_MANIFEST, METRICS_DIR
from src.data.cache_maps import cache_paths
from src.data.plot_utils import CLASSES, display
from src.features.frozen import load_frozen_embeddings, read_manifest
from src.models import embed_eval
from src.models.dataset import split_index
from src.models.metrics import OK_INDEX

K = 5
SIM_TOLERANCE = 1e-5  # float32 rounding between FAISS and torch
MACRO_TOLERANCE = 1e-3  # a tie at the k-th neighbor can swap its label


def normalize(x: np.ndarray) -> np.ndarray:
    """Float32 copy of the rows scaled to length 1 (cosine becomes inner product)."""
    x = np.array(x, dtype=np.float32, order="C")
    faiss.normalize_L2(x)
    return x


@dataclass
class Gallery:
    """Exact cosine index over gallery embeddings, with their labels and wafer ids."""

    index: faiss.Index
    labels: np.ndarray
    wafer_ids: np.ndarray

    @classmethod
    def build(
        cls, embeddings: np.ndarray, labels: np.ndarray, wafer_ids: np.ndarray
    ) -> "Gallery":
        """Index every gallery row."""
        if not len(embeddings) == len(labels) == len(wafer_ids):
            raise ValueError(
                "Embeddings, labels and wafer ids must have the same length"
            )
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(normalize(embeddings))
        return cls(index, np.asarray(labels), np.asarray(wafer_ids))

    def __len__(self) -> int:
        return self.index.ntotal

    def search(self, queries: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
        """Cosine similarity and gallery row of the k nearest rows for every query."""
        if not 1 <= k <= len(self):
            raise ValueError(f"k must be between 1 and the gallery size ({len(self)})")
        return self.index.search(normalize(np.atleast_2d(queries)), k)

    def neighbors(self, query: np.ndarray, k: int = K) -> list[dict]:
        """The k nearest wafers to one query, as readable records."""
        sims, rows = self.search(query, k)
        return [
            {
                "wafer_id": int(self.wafer_ids[r]),
                "label": display(CLASSES[self.labels[r]]),
                "similarity": round(float(s), 4),
            }
            for s, r in zip(sims[0], rows[0])
        ]


def precision_at_k(
    gallery: Gallery, queries: np.ndarray, query_labels: np.ndarray, k: int = K
) -> dict:
    """Share of the k neighbors with the query's defect type, for defect queries only."""
    defect = query_labels != OK_INDEX
    _, rows = gallery.search(queries[defect], k)
    hits = gallery.labels[rows] == query_labels[defect][:, None]
    per_class = {
        display(CLASSES[c]): round(float(hits[query_labels[defect] == c].mean()), 4)
        for c in np.unique(query_labels[defect])
    }
    return {
        "k": k,
        "macro": float(np.mean(list(per_class.values()))),
        "per_class": per_class,
        "hit_rate": float(hits.any(axis=1).mean()),
    }


def train_wafer_ids(size: int = 64) -> np.ndarray:
    """Wafer ids of the labeled train wafers, in the order of the frozen embeddings."""
    rows, _ = split_index("train", size)
    _, ids_path = cache_paths(size, "labeled")
    return np.load(ids_path)[rows]


def frozen_gallery() -> Gallery:
    """Gallery of all labeled train wafers from the frozen model."""
    x, y = load_frozen_embeddings("train")
    ids = train_wafer_ids()
    if len(ids) != len(y):
        raise ValueError("Wafer ids do not match the frozen train embeddings")
    return Gallery.build(x, y, ids)


def max_similarity_gap(
    gallery: Gallery, gallery_x: np.ndarray, queries: np.ndarray, k: int = K
) -> float:
    """Largest difference between FAISS and brute-force top-k similarities.

    Wafers at exactly the same distance may be returned in either order, so the
    two searches are compared by their similarities, not by the rows they return.
    """
    faiss_sims, _ = gallery.search(queries, k)
    brute_sims, _ = embed_eval.nearest(queries, gallery_x, k)
    return float(np.abs(faiss_sims - brute_sims).max())


def main() -> None:
    """Build the gallery, query every val defect wafer, compare with brute-force search."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=K)
    args = p.parse_args()

    start = time.time()
    gallery = frozen_gallery()
    build_s = time.time() - start
    val_x, val_y = load_frozen_embeddings("val")

    start = time.time()
    result = precision_at_k(gallery, val_x, val_y, args.k)
    query_ms = (time.time() - start) * 1000 / int((val_y != OK_INDEX).sum())

    train_x = load_frozen_embeddings("train")[0]
    brute = embed_eval.similar_cases((train_x, gallery.labels), (val_x, val_y), args.k)
    sim_gap = max_similarity_gap(gallery, train_x, val_x[val_y != OK_INDEX], args.k)
    if (
        sim_gap > SIM_TOLERANCE
        or abs(result["macro"] - brute["macro"]) > MACRO_TOLERANCE
    ):
        raise ValueError(
            f"FAISS {result['macro']:.4f} differs from brute force {brute['macro']:.4f}"
            f" (largest similarity gap {sim_gap:.2e})"
        )

    out = {
        "split": "val",
        "model": read_manifest(FROZEN_MANIFEST)["name"],
        "gallery": len(gallery),
        "gallery_split": "train (labeled)",
        "queries": int((val_y != OK_INDEX).sum()),
        "brute_force_macro": brute["macro"],
        "max_similarity_gap": sim_gap,
        "build_seconds": round(build_s, 2),
        "ms_per_query": round(query_ms, 3),
        **result,
    }
    (METRICS_DIR / "retrieval_val.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
