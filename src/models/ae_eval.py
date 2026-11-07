"""Anomaly scores and evaluation for the autoencoder.

The score is computed on the defect channel only, inside the wafer: errors in the
wafer-area channel only reflect wafer shape, not defects. O.K. wafers also have
scattered single failed dies that no model can predict, so three scores are tried:
  mean     - mean error over the wafer
  top1     - mean of the 1% of pixels with the largest error
  blur_max - largest error after a 5x5 blur
  density  - largest local defect density that the rebuild does not explain: blur the input
             and the rebuild separately, then subtract. Scattered single dies blur to a low
             density, close to what the model predicts; an unexplained cluster stays high.
The task is defect vs O.K.; the best score is chosen on val by AUROC.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.plot_utils import CLASSES, display, save, setup_style
from src.models.metrics import OK_INDEX

SCORES = ("mean", "top1", "blur_max", "density")
BLUR = 5
DENSITY_BLUR = 11  # best of 3-15 on val


def blur(img: torch.Tensor, size: int) -> torch.Tensor:
    """Mean over a size x size window around each pixel (N x H x W)."""
    return F.avg_pool2d(img[:, None], size, stride=1, padding=size // 2)[:, 0]


def error_map(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Per-pixel rebuild error of the defect channel, zero outside the wafer (N x H x W)."""
    rebuilt = torch.sigmoid(model(x))
    return (rebuilt[:, 1] - x[:, 1]).abs() * x[:, 0]


def density_map(model: torch.nn.Module, x: torch.Tensor, size: int = DENSITY_BLUR) -> torch.Tensor:
    """Local defect density of the input minus that of the rebuild (N x H x W)."""
    rebuilt = torch.sigmoid(model(x))[:, 1] * x[:, 0]
    return blur(x[:, 1], size) - blur(rebuilt, size)


def batch_scores(model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """The anomaly scores for a batch of input maps."""
    area = x[:, 0]
    rebuilt = torch.sigmoid(model(x))[:, 1] * area
    err = (rebuilt - x[:, 1]).abs() * area
    flat = err.flatten(1)
    k = max(1, int(0.01 * flat.shape[1]))
    density = blur(x[:, 1], DENSITY_BLUR) - blur(rebuilt, DENSITY_BLUR)
    return {
        "mean": flat.sum(1) / area.flatten(1).sum(1).clamp(min=1),
        "top1": flat.topk(k, dim=1).values.mean(1),
        "blur_max": blur(err, BLUR).flatten(1).amax(1),
        "density": density.flatten(1).amax(1),
    }


@torch.no_grad()
def compute_scores(model: torch.nn.Module, loader: DataLoader,
                   device: torch.device) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """True labels and every anomaly score for a whole loader."""
    model.eval()
    ys, out = [], {s: [] for s in SCORES}
    for x, y in loader:
        x = x.to(device)
        for name, value in batch_scores(model, x).items():
            out[name].append(value.cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), {s: torch.cat(v).numpy() for s, v in out.items()}


def operating_point(score: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Defect vs O.K. metrics when every wafer above the threshold is flagged."""
    defect, flagged = y != OK_INDEX, score > threshold
    tp, fp = int((defect & flagged).sum()), int((~defect & flagged).sum())
    fn, tn = int((defect & ~flagged).sum()), int((~defect & ~flagged).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    caught = {display(CLASSES[c]): round(float(flagged[y == c].mean()), 4)
              for c in range(len(CLASSES)) if c != OK_INDEX and (y == c).any()}
    return {"threshold": float(threshold),
            "missed_defect_rate": fn / max(tp + fn, 1), "false_alarm_rate": fp / max(fp + tn, 1),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "confusion_matrix": [[tn, fp], [fn, tp]], "caught_per_class": caught}


def evaluate_scores(y: np.ndarray, scores: dict[str, np.ndarray],
                    false_alarm: float, max_missed: float) -> dict:
    """AUROC of every score; two operating points for the best one.

    at_false_alarm: threshold that flags `false_alarm` of O.K. wafers (the CNN's rate).
    at_max_missed:  threshold that misses `max_missed` of defects (the selection rule).
    """
    defect = y != OK_INDEX
    auroc = {s: float(roc_auc_score(defect, v)) for s, v in scores.items()}
    best = max(auroc, key=auroc.get)
    score = scores[best]
    return {
        "auroc": auroc, "best_score": best,
        "at_false_alarm": operating_point(score, y, np.quantile(score[~defect], 1 - false_alarm)),
        "at_max_missed": operating_point(score, y, np.quantile(score[defect], max_missed)),
    }


def summary_line(result: dict) -> str:
    """One line for logs: AUROC per score and the missed defects at the CNN false-alarm rate."""
    auroc = " ".join(f"{s} {v:.3f}" for s, v in result["auroc"].items())
    op = result["at_false_alarm"]
    return (f"AUROC {auroc} | {result['best_score']}: missed {op['missed_defect_rate']:.1%} "
            f"at false alarms {op['false_alarm_rate']:.1%}")


def plot_binary_confusion(cm: list[list[int]], title: str, filename: str) -> None:
    """2x2 row-normalized confusion matrix: O.K. vs defect."""
    setup_style()
    cm = np.asarray(cm, dtype=float)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    names = ["O.K.", "Defect"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = "white" if share[i, j] > 0.5 else "black"
            ax.text(j, i, f"{share[i, j]:.1%}\n{int(cm[i, j]):,}", ha="center", va="center",
                    color=color, fontsize=18)
    ax.set_xticks(range(2), names)
    ax.set_yticks(range(2), names)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    save(fig, filename)


def plot_scores_by_class(score: np.ndarray, y: np.ndarray, threshold: float,
                         title: str, filename: str) -> None:
    """Box plot of the anomaly score per true class, with the decision threshold."""
    setup_style()
    names = [display(c) for c in CLASSES]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.boxplot([score[y == c] for c in range(len(CLASSES))], showfliers=False)
    ax.axhline(threshold, color="darkorange", linestyle="--", label="Threshold")
    ax.set_xticks(range(1, len(names) + 1), names, rotation=45, ha="right")
    ax.set(ylabel="Anomaly score", title=title)
    ax.legend()
    save(fig, filename)


@torch.no_grad()
def plot_examples(model: torch.nn.Module, dataset, score: np.ndarray, title: str,
                  filename: str, device: torch.device, score_name: str = "top1",
                  classes: tuple[str, ...] = ("none", "Center", "Edge-Ring", "Loc", "Scratch", "Random")) -> None:
    """For a typical wafer of each class: input, rebuilt map, and the map behind the score."""
    setup_style()
    model.eval()
    labels = np.asarray(dataset.labels)
    fig, axes = plt.subplots(len(classes), 3, figsize=(10, 3.4 * len(classes)))
    for row, cls in zip(axes, classes):
        idx = np.flatnonzero(labels == CLASSES.index(cls))
        i = idx[np.argsort(score[idx])[len(idx) // 2]]  # wafer with the median score of its class
        x = dataset[i][0][None].to(device)
        rebuilt = torch.sigmoid(model(x))[0].cpu().numpy()
        x = x[0].cpu().numpy()
        xt = torch.from_numpy(x)[None].to(device)
        if score_name == "density":
            err, label = density_map(model, xt)[0].clamp(min=0).cpu().numpy(), "Unexplained density"
        else:
            err, label = error_map(model, xt)[0].cpu().numpy(), "Error"
        panels = [(x[0] + x[1], "viridis", 2, f"{display(cls)}: input"),
                  (rebuilt[0] * x[0] + rebuilt[1] * x[0], "viridis", 2, "Rebuilt"),
                  (err, "magma", 1, f"{label} ({score[i]:.3f})")]
        for ax, (img, cmap, vmax, name) in zip(row, panels):
            ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(name, fontsize=16)
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, filename)
