"""New-type scores: how unlike the known defect types a wafer looks (higher = newer).

All four scores use only what a deployed model would have: the known classes'
logits and the train embeddings of known classes.
  softmax     - 1 minus the highest class probability (an unsure model)
  energy      - minus logsumexp of the logits (a weak response to every class)
  knn         - mean cosine distance to the k nearest train embeddings
  mahalanobis - distance to the nearest class mean, scaled by the shared
                within-class covariance (unusual directions count more)
"""

import numpy as np
import torch

from src.models.embed_eval import knn_distance

MAHALANOBIS_RIDGE = 1e-3  # added to the covariance diagonal, as a share of its mean
SCORES = ("softmax", "energy", "knn", "mahalanobis")


def logits(emb: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Class scores from embeddings and the CNN's linear head (dropout is off at eval)."""
    return emb @ weight.T + bias


def softmax_score(logit: np.ndarray, known: np.ndarray) -> np.ndarray:
    """1 - max probability over the known classes only."""
    probs = torch.softmax(torch.from_numpy(logit[:, known]).double(), dim=1)
    return 1 - probs.max(dim=1).values.numpy()


def energy_score(logit: np.ndarray, known: np.ndarray) -> np.ndarray:
    """-logsumexp of the known classes' logits."""
    return -torch.logsumexp(torch.from_numpy(logit[:, known]).double(), dim=1).numpy()


def fit_mahalanobis(
    emb: np.ndarray, y: np.ndarray, classes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Class means and the inverse of the pooled within-class covariance."""
    means = np.stack([emb[y == c].mean(axis=0) for c in classes])
    centered = np.concatenate([emb[y == c] - m for c, m in zip(classes, means)])
    cov = centered.T @ centered / len(centered)
    cov += MAHALANOBIS_RIDGE * np.trace(cov) / len(cov) * np.eye(len(cov))
    return means, np.linalg.inv(cov)


def mahalanobis_score(
    emb: np.ndarray, means: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    """Squared Mahalanobis distance to the nearest class mean."""
    diff = emb[:, None, :] - means[None, :, :]
    return np.einsum("nci,ij,ncj->nc", diff, precision, diff).min(axis=1)


def all_scores(
    train: tuple, val_emb: np.ndarray, head: tuple, known: np.ndarray, k: int = 5
) -> dict[str, np.ndarray]:
    """The four scores of every val wafer; train holds (embeddings, labels) of known classes."""
    train_emb, train_y = train
    logit = logits(val_emb, *head)
    means, precision = fit_mahalanobis(train_emb, train_y, known)
    return {
        "softmax": softmax_score(logit, known),
        "energy": energy_score(logit, known),
        "knn": knn_distance(val_emb, train_emb, k),
        "mahalanobis": mahalanobis_score(val_emb, means, precision),
    }
