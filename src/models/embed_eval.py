"""Tests of a wafer embedding, one per feature that will use it.

All distances are cosine distances between embeddings (1 - cosine similarity).
  similar cases  - precision@k: share of a val defect wafer's k nearest train wafers
                   that have the same defect type
  anomaly        - score = mean distance to the k nearest O.K. train wafers
  new type       - one defect type is left out of the train gallery; score = mean
                   distance to the k nearest train wafers of the known types
  few labels     - linear probe (logistic regression) on the embedding, trained on
                   a share of the train labels; F1 macro on val
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.data.plot_utils import CLASSES, display
from src.models.dataset import class_weights
from src.models.metrics import OK_INDEX, evaluate


@torch.no_grad()
def extract(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings (model.embed) and labels for a whole labeled loader."""
    model.eval()
    embs, ys = [], []
    for x, y in loader:
        embs.append(model.embed(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(embs).numpy(), torch.cat(ys).numpy()


def nearest(query: np.ndarray, gallery: np.ndarray, k: int, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity and index of the k nearest gallery rows for every query row."""
    g = torch.nn.functional.normalize(torch.from_numpy(gallery), dim=1)
    sims, idx = [], []
    for start in range(0, len(query), chunk):
        q = torch.nn.functional.normalize(torch.from_numpy(query[start:start + chunk]), dim=1)
        s, i = (q @ g.T).topk(k, dim=1)
        sims.append(s)
        idx.append(i)
    return torch.cat(sims).numpy(), torch.cat(idx).numpy()


def knn_distance(query: np.ndarray, gallery: np.ndarray, k: int) -> np.ndarray:
    """Mean cosine distance to the k nearest gallery rows (higher = more unusual)."""
    sims, _ = nearest(query, gallery, k)
    return 1 - sims.mean(axis=1)


def similar_cases(train: tuple, val: tuple, k: int = 5) -> dict:
    """precision@k for val defect wafers searched among all labeled train wafers."""
    (g_emb, g_y), (q_emb, q_y) = train, val
    defect = q_y != OK_INDEX
    _, idx = nearest(q_emb[defect], g_emb, k)
    hit = (g_y[idx] == q_y[defect][:, None]).mean(axis=1)
    per_class = {display(CLASSES[c]): round(float(hit[q_y[defect] == c].mean()), 4)
                 for c in np.unique(q_y[defect])}
    return {"k": k, "macro": float(np.mean(list(per_class.values()))), "per_class": per_class}


def anomaly(train: tuple, val: tuple, k: int = 5) -> np.ndarray:
    """Anomaly score of every val wafer: distance to the nearest O.K. train wafers."""
    (g_emb, g_y), (q_emb, _) = train, val
    return knn_distance(q_emb, g_emb[g_y == OK_INDEX], k)


def new_type(train: tuple, val: tuple, holdout: str, k: int = 5, false_alarm: float = 0.05) -> dict:
    """How well a left-out defect type stands out from the known types.

    auroc_vs_known_defects: held-out wafers vs val wafers of the other defect types.
    flagged_as_new: share of held-out wafers above the threshold that flags
    `false_alarm` of the known val wafers (O.K. and defects) as new.
    """
    (g_emb, g_y), (q_emb, q_y) = train, val
    h = CLASSES.index(holdout)
    score = knn_distance(q_emb, g_emb[g_y != h], k)
    new, known_defect = q_y == h, (q_y != h) & (q_y != OK_INDEX)
    threshold = np.quantile(score[~new], 1 - false_alarm)
    pick = new | known_defect
    return {"holdout": holdout, "k": k,
            "auroc_vs_known_defects": float(roc_auc_score(new[pick], score[pick])),
            "auroc_vs_all_known": float(roc_auc_score(new, score)),
            "flagged_as_new": float((score[new] > threshold).mean()), "false_alarm": false_alarm}


def sample_labels(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Row indices: `frac` of each class, at least one wafer per class."""
    rng = np.random.default_rng(seed)
    rows = [rng.permutation(np.flatnonzero(y == c))[:max(1, round(frac * (y == c).sum()))]
            for c in np.unique(y)]
    return np.concatenate(rows)


def probe(train: tuple, val: tuple, frac: float, seed: int = 0) -> dict:
    """Linear classifier on the embedding, trained on `frac` of the train labels.

    Class weights 1/sqrt(count), as in the CNN ('balanced' = 1/count flags too many O.K. wafers).
    """
    (x, y), (vx, vy) = train, val
    rows = sample_labels(y, frac, seed)
    weights = class_weights(y[rows]).numpy()
    scaler = StandardScaler().fit(x[rows])
    clf = LogisticRegression(max_iter=2000, class_weight={c: float(weights[c]) for c in np.unique(y)})
    clf.fit(scaler.transform(x[rows]), y[rows])
    return {"frac": frac, "n_train": int(len(rows)), **evaluate(vy, clf.predict(scaler.transform(vx)))}
