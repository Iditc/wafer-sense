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
import pandas as pd

from src.config import FROZEN_MANIFEST, METRICS_DIR, WAFERS_PARQUET
from src.data.cache_maps import cache_paths
from src.data.plot_utils import CLASSES, display
from src.features.frozen import load_frozen_embeddings, read_manifest
from src.models import embed_eval
from src.models.dataset import split_index
from src.models.metrics import OK_INDEX

# Chosen on day 3 (val): precision barely drops from k=1 to 5 (0.872 -> 0.866) while the
# share of queries with a same-type neighbor rises from 89% to 96%. One neighbor per lot
# costs no precision (0.865) and always shows 5 different lots instead of 4.5 on average.
K = 5
MAX_PER_LOT = 1
SIM_TOLERANCE = 1e-5  # float32 rounding between FAISS and torch
MACRO_TOLERANCE = 1e-3  # a tie at the k-th neighbor can swap its label
FETCH_FACTOR = 10  # per-lot search: candidates fetched per result, grown if too few fit


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
    lots: np.ndarray | None = None

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        wafer_ids: np.ndarray,
        lots: np.ndarray | None = None,
    ) -> "Gallery":
        """Index every gallery row; lots are optional (needed for the per-lot limit)."""
        sizes = {len(embeddings), len(labels), len(wafer_ids)}
        if lots is not None:
            sizes.add(len(lots))
        if len(sizes) != 1:
            raise ValueError(
                "Embeddings, labels, wafer ids and lots must have the same length"
            )
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(normalize(embeddings))
        lots = None if lots is None else np.asarray(lots)
        return cls(index, np.asarray(labels), np.asarray(wafer_ids), lots)

    def __len__(self) -> int:
        return self.index.ntotal

    def search(self, queries: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
        """Cosine similarity and gallery row of the k nearest rows for every query."""
        if not 1 <= k <= len(self):
            raise ValueError(f"k must be between 1 and the gallery size ({len(self)})")
        return self.index.search(normalize(np.atleast_2d(queries)), k)

    def search_per_lot(
        self, queries: np.ndarray, k: int = K, max_per_lot: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Like search, but at most max_per_lot neighbors come from one lot.

        Wafers of one lot are often near-copies; the limit spreads the k results
        over more lots. Candidates are fetched in growing batches until k fit.
        """
        if self.lots is None:
            raise ValueError("The gallery has no lots; build it with lots=...")
        queries = np.atleast_2d(queries)
        sims = np.empty((len(queries), k), dtype=np.float32)
        rows = np.empty((len(queries), k), dtype=np.int64)
        for i, query in enumerate(queries):
            fetch = min(len(self), k * FETCH_FACTOR)
            while True:
                cand_sims, cand_rows = self.search(query, fetch)
                keep = limit_per_lot(cand_rows[0], self.lots, k, max_per_lot)
                if len(keep) == k or fetch == len(self):
                    break
                fetch = min(len(self), fetch * FETCH_FACTOR)
            if len(keep) < k:
                raise ValueError(f"Only {len(keep)} results fit the per-lot limit")
            sims[i], rows[i] = cand_sims[0][keep], cand_rows[0][keep]
        return sims, rows

    def neighbors(
        self, query: np.ndarray, k: int = K, max_per_lot: int | None = MAX_PER_LOT
    ) -> list[dict]:
        """The k nearest wafers to one query, as readable records.

        Uses the per-lot limit when the gallery knows its lots; None turns it off.
        """
        if max_per_lot is None or self.lots is None:
            sims, rows = self.search(query, k)
        else:
            sims, rows = self.search_per_lot(query, k, max_per_lot)
        records = []
        for s, r in zip(sims[0], rows[0]):
            record = {
                "wafer_id": int(self.wafer_ids[r]),
                "label": display(CLASSES[self.labels[r]]),
                "similarity": round(float(s), 4),
            }
            if self.lots is not None:
                record["lot"] = str(self.lots[r])
            records.append(record)
        return records


def limit_per_lot(
    candidates: np.ndarray, lots: np.ndarray, k: int, max_per_lot: int
) -> np.ndarray:
    """Positions of the first k candidates (in order) with at most max_per_lot per lot."""
    if max_per_lot < 1:
        raise ValueError("max_per_lot must be at least 1")
    counts: dict = {}
    keep = []
    for pos, row in enumerate(candidates):
        lot = lots[row]
        if counts.get(lot, 0) < max_per_lot:
            counts[lot] = counts.get(lot, 0) + 1
            keep.append(pos)
            if len(keep) == k:
                break
    return np.array(keep, dtype=np.int64)


def precision_at_k(
    gallery: Gallery,
    queries: np.ndarray,
    query_labels: np.ndarray,
    k: int = K,
    max_per_lot: int | None = None,
) -> dict:
    """Share of the k neighbors with the query's defect type, for defect queries only."""
    defect = query_labels != OK_INDEX
    if max_per_lot is None:
        _, rows = gallery.search(queries[defect], k)
    else:
        _, rows = gallery.search_per_lot(queries[defect], k, max_per_lot)
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


def split_wafer_ids(split: str, size: int = 64) -> np.ndarray:
    """Wafer ids of a split's labeled wafers, in the order of the frozen embeddings."""
    rows, _ = split_index(split, size)
    _, ids_path = cache_paths(size, "labeled")
    return np.load(ids_path)[rows]


def train_wafer_ids(size: int = 64) -> np.ndarray:
    """Wafer ids of the labeled train wafers, in the order of the frozen embeddings."""
    return split_wafer_ids("train", size)


def lot_names(wafer_ids: np.ndarray) -> np.ndarray:
    """Lot name of every wafer id, in the same order."""
    lots = pd.read_parquet(WAFERS_PARQUET, columns=["wafer_id", "lot_name"])
    return lots.set_index("wafer_id").loc[wafer_ids, "lot_name"].to_numpy()


def frozen_gallery() -> Gallery:
    """Gallery of all labeled train wafers from the frozen model, with their lots."""
    x, y = load_frozen_embeddings("train")
    ids = train_wafer_ids()
    if len(ids) != len(y):
        raise ValueError("Wafer ids do not match the frozen train embeddings")
    return Gallery.build(x, y, ids, lot_names(ids))


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
