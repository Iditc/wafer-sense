"""Day 5: compare new-type scores on the 8 leave-one-type-out CNNs (val only).

For each run, val wafers of the held-out type are "new"; val wafers of the other
defect types are "known". A score is good if new wafers score higher (AUROC).
The threshold is set on known defects only: it flags FLAG_RATE of them as new,
because the new-type check runs only on wafers already called defective.
The macro AUROC gap to kNN (the day 4 score) comes with a bootstrap 95% interval.
Run: python -m src.openset.compare_scores  (about an hour on CPU the first time)
"""

import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_DIR, METRICS_DIR, PROCESSED_DIR
from src.data.plot_utils import CLASSES
from src.features.frozen import load_frozen_embeddings
from src.models import embed_eval as ee
from src.models.cnn import WaferCNN
from src.models.dataset import WaferDataset
from src.models.metrics import OK_INDEX
from src.openset.holdout import holdout_runs
from src.openset.plots import plot_score_auroc
from src.openset.scores import SCORES, all_scores

FLAG_RATE = 0.05  # share of known defects flagged as new
BASELINE = "knn"
BOOTSTRAP = 1000
SEED = 0


def load_head(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Weight and bias of a checkpoint's linear class head."""
    state = torch.load(CHECKPOINT_DIR / f"{name}.pt", map_location="cpu")
    return state["head.weight"].numpy(), state["head.bias"].numpy()


def run_embeddings(name: str) -> dict[str, tuple]:
    """Train and val (embedding, label) pairs of a run, extracted once and cached."""
    path = PROCESSED_DIR / f"emb_{name}.npz"
    if not path.exists():
        model = WaferCNN()
        model.load_state_dict(
            torch.load(CHECKPOINT_DIR / f"{name}.pt", map_location="cpu")
        )
        out = {}
        for split in ("train", "val"):
            loader = DataLoader(WaferDataset(split), batch_size=512, num_workers=0)
            with torch.no_grad():
                out[f"{split}_x"], out[f"{split}_y"] = ee.extract(
                    model, loader, torch.device("cpu")
                )
            print(f"{name} {split}: {out[f'{split}_x'].shape}", flush=True)
        np.savez(path, **out)
    with np.load(path) as data:
        return {s: (data[f"{s}_x"], data[f"{s}_y"]) for s in ("train", "val")}


def threshold(known_scores: np.ndarray, rate: float = FLAG_RATE) -> float:
    """Score above which `rate` of the known defects lie."""
    return float(np.quantile(known_scores, 1 - rate))


def evaluate_run(name: str, defect: str) -> dict:
    """Scores of one held-out run on its val defects, with AUROC and flagged share."""
    emb = run_embeddings(name)
    h = CLASSES.index(defect)
    train_x, train_y = emb["train"]
    val_x, val_y = emb["val"]
    known = np.array([c for c in range(len(CLASSES)) if c != h])
    defects = val_y != OK_INDEX
    scores = all_scores(
        (train_x[train_y != h], train_y[train_y != h]),
        val_x[defects],
        load_head(name),
        known,
    )
    new = val_y[defects] == h
    out = {"defect": defect, "new": new, "scores": scores, "results": {}}
    for s in SCORES:
        cut = threshold(scores[s][~new])
        out["results"][s] = {
            "auroc": round(float(roc_auc_score(new, scores[s])), 4),
            "flagged_as_new": round(float((scores[s][new] > cut).mean()), 4),
        }
    return out


def bootstrap_gap(runs: list[dict], score: str, seed: int = SEED) -> list[float]:
    """95% interval of macro AUROC(score) - macro AUROC(baseline), resampling each run's wafers."""
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(BOOTSTRAP):
        diffs = []
        for run in runs:
            pick = rng.integers(0, len(run["new"]), len(run["new"]))
            new = run["new"][pick]
            diffs.append(
                roc_auc_score(new, run["scores"][score][pick])
                - roc_auc_score(new, run["scores"][BASELINE][pick])
            )
        gaps.append(np.mean(diffs))
    return [round(float(v), 4) for v in np.percentile(gaps, [2.5, 97.5])]


def frozen_thresholds() -> dict[str, float]:
    """Thresholds of every score for the deployed model, cnn_cosine, on its val defects."""
    train, val = load_frozen_embeddings("train"), load_frozen_embeddings("val")
    defects = val[1] != OK_INDEX
    scores = all_scores(
        train, val[0][defects], load_head("cnn_cosine"), np.arange(len(CLASSES))
    )
    return {s: round(threshold(v), 6) for s, v in scores.items()}


def summarize(runs: list[dict]) -> dict:
    """JSON-ready table: per score, the macro and per-type results and the gap to kNN."""
    table = {}
    for s in SCORES:
        per_type = {r["defect"]: r["results"][s] for r in runs}
        table[s] = {
            "macro_auroc": round(
                float(np.mean([v["auroc"] for v in per_type.values()])), 4
            ),
            "macro_flagged_as_new": round(
                float(np.mean([v["flagged_as_new"] for v in per_type.values()])), 4
            ),
            "gap_to_knn_ci": [0.0, 0.0] if s == BASELINE else bootstrap_gap(runs, s),
            "per_type": per_type,
        }
        print(s, {k: v for k, v in table[s].items() if k != "per_type"}, flush=True)
    return table


def main() -> None:
    """Score all 8 runs, save the comparison and the frozen model's thresholds."""
    torch.set_num_threads(os.cpu_count() or 1)
    runs = [
        evaluate_run(name, defect) for name, defect in holdout_runs(skip=()).items()
    ]
    out = {
        "split": "val",
        "flag_rate": FLAG_RATE,
        "baseline": BASELINE,
        "scores": summarize(runs),
        "frozen_thresholds": frozen_thresholds(),
    }
    (METRICS_DIR / "openset_scores_val.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    plot_score_auroc(out["scores"])


if __name__ == "__main__":
    main()
