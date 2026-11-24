"""The frozen model: one checkpoint and one set of embeddings for every feature.

The manifest records the SHA-256 of both files and the val results. Loading fails
if a file changed, so a feature can never silently run on different weights.
Only train and val embeddings exist here; the test set stays closed.
Run once: python -m src.features.frozen
"""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (
    CHECKPOINT_DIR,
    FROZEN_MANIFEST,
    FROZEN_MODEL,
    METRICS_DIR,
    PROCESSED_DIR,
)
from src.models.cnn import WaferCNN
from src.models.dataset import WaferDataset
from src.models.metrics import evaluate
from src.models.train_utils import get_device

EMBEDDING_SPLITS = ("train", "val")
METRIC_KEYS = ("f1_macro", "accuracy", "missed_defect_rate", "false_alarm_rate")
METRIC_TOLERANCE = (
    0.001  # GPU and CPU can round a few borderline predictions differently
)
MIN_COSINE = 0.999
TRAIN_CHECK_SIZE = 2000


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    """Fail if a file is missing or differs from the frozen version."""
    if not path.exists():
        raise FileNotFoundError(f"Frozen file not found: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{path.name} changed: expected {expected[:12]}, found {actual[:12]}"
        )


def read_manifest(path: Path = FROZEN_MANIFEST) -> dict:
    """The frozen model's record."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_model(
    device: torch.device | str = "cpu",
    manifest_path: Path = FROZEN_MANIFEST,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> WaferCNN:
    """The verified model in eval mode, with gradients off."""
    entry = read_manifest(manifest_path)["checkpoint"]
    path = checkpoint_dir / entry["file"]
    verify_file(path, entry["sha256"])
    model = WaferCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.requires_grad_(False)
    return model.to(device).eval()


def load_frozen_embeddings(
    split: str,
    manifest_path: Path = FROZEN_MANIFEST,
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Verified embeddings and labels of train or val."""
    if split not in EMBEDDING_SPLITS:
        raise ValueError(
            f"No frozen embeddings for '{split}': use one of {EMBEDDING_SPLITS}"
        )
    entry = read_manifest(manifest_path)["embeddings"]
    path = processed_dir / entry["file"]
    verify_file(path, entry["sha256"])
    with np.load(path) as data:
        return data[f"{split}_x"], data[f"{split}_y"]


@torch.no_grad()
def embed_and_predict(
    model: WaferCNN, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embeddings, predicted classes and true labels for a whole loader."""
    model.eval()
    embeddings, predictions, labels = [], [], []
    for x, y in loader:
        h = model.embed(x.to(device))
        embeddings.append(h.cpu())
        predictions.append(model.head(h).argmax(dim=1).cpu())
        labels.append(y)
    return (
        torch.cat(embeddings).numpy(),
        torch.cat(predictions).numpy(),
        torch.cat(labels).numpy(),
    )


def min_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Lowest cosine similarity between matching rows."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return float((a_norm * b_norm).sum(axis=1).min())


def check_metrics(
    new: dict, recorded: dict, tolerance: float = METRIC_TOLERANCE
) -> None:
    """Fail if the model no longer reaches its recorded results."""
    for key in METRIC_KEYS:
        if abs(new[key] - recorded[key]) > tolerance:
            raise ValueError(f"{key}: got {new[key]:.4f}, recorded {recorded[key]:.4f}")


def check_embeddings(
    new: np.ndarray, saved: np.ndarray, threshold: float = MIN_COSINE
) -> float:
    """Fail if recomputed embeddings differ from the saved ones; returns the lowest cosine."""
    if new.shape != saved.shape:
        raise ValueError(f"Shape mismatch: {new.shape} vs {saved.shape}")
    lowest = min_cosine(new, saved)
    if lowest < threshold:
        raise ValueError(f"Embeddings differ: lowest cosine {lowest:.6f}")
    return lowest


def check_can_write(path: Path, checkpoint_sha: str, force: bool) -> None:
    """Refuse to replace a manifest of a different checkpoint unless asked."""
    if (
        path.exists()
        and not force
        and read_manifest(path)["checkpoint"]["sha256"] != checkpoint_sha
    ):
        raise FileExistsError(
            f"{path.name} records another checkpoint; use --force to replace it"
        )


def verify_and_build(name: str, device: torch.device, workers: int) -> dict:
    """Recompute val and a train sample, compare with saved results, return the manifest."""
    checkpoint = CHECKPOINT_DIR / f"{name}.pt"
    embeddings = PROCESSED_DIR / f"emb_{name}.npz"
    model = WaferCNN()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    saved = np.load(embeddings)
    recorded = json.loads((METRICS_DIR / f"{name}.json").read_text(encoding="utf-8"))

    val_loader = DataLoader(WaferDataset("val"), batch_size=512, num_workers=workers)
    val_x, val_pred, val_y = embed_and_predict(model, val_loader, device)
    if not np.array_equal(val_y, saved["val_y"]):
        raise ValueError(
            "Val labels are in a different order than the saved embeddings"
        )
    metrics = evaluate(val_y, val_pred)
    check_metrics(metrics, recorded)
    val_cosine = check_embeddings(val_x, saved["val_x"])

    pick = np.linspace(0, len(saved["train_y"]) - 1, TRAIN_CHECK_SIZE).astype(int)
    train_loader = DataLoader(
        WaferDataset("train").keep(pick), batch_size=512, num_workers=workers
    )
    train_x, _, train_y = embed_and_predict(model, train_loader, device)
    if not np.array_equal(train_y, saved["train_y"][pick]):
        raise ValueError(
            "Train labels are in a different order than the saved embeddings"
        )
    train_cosine = check_embeddings(train_x, saved["train_x"][pick])

    return {
        "name": name,
        "frozen_on": date.today().isoformat(),
        "checkpoint": {"file": checkpoint.name, "sha256": file_sha256(checkpoint)},
        "embeddings": {
            "file": embeddings.name,
            "sha256": file_sha256(embeddings),
            "dim": int(saved["val_x"].shape[1]),
            "n_train": int(len(saved["train_y"])),
            "n_val": int(len(saved["val_y"])),
        },
        "val": {key: metrics[key] for key in METRIC_KEYS},
        "recorded_val": {key: recorded[key] for key in METRIC_KEYS},
        "lowest_cosine": {"val": val_cosine, "train_sample": train_cosine},
    }


def main() -> None:
    """Verify the chosen model and write its manifest."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default=FROZEN_MODEL)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--force", action="store_true", help="replace a manifest of another checkpoint"
    )
    args = p.parse_args()
    check_can_write(
        FROZEN_MANIFEST, file_sha256(CHECKPOINT_DIR / f"{args.name}.pt"), args.force
    )
    manifest = verify_and_build(args.name, get_device(), args.workers)
    FROZEN_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
