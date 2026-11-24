"""Tests for the frozen model: changed files must fail, recorded results must hold."""

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import FROZEN_MANIFEST, FROZEN_MODEL
from src.features import frozen
from src.models.cnn import WaferCNN
from src.models.dataset import WaferDataset
from src.models.metrics import evaluate


@pytest.fixture
def frozen_files(tmp_path):
    """A small checkpoint, embeddings file and manifest in a temp folder."""
    model = WaferCNN()
    checkpoint = tmp_path / "tiny.pt"
    embeddings = tmp_path / "emb_tiny.npz"
    torch.save(model.state_dict(), checkpoint)
    np.savez(
        embeddings,
        train_x=np.ones((4, 256), dtype=np.float32),
        train_y=np.arange(4),
        val_x=np.ones((2, 256), dtype=np.float32),
        val_y=np.arange(2),
    )
    manifest = tmp_path / "manifest.json"
    record = {
        "checkpoint": {
            "file": checkpoint.name,
            "sha256": frozen.file_sha256(checkpoint),
        },
        "embeddings": {
            "file": embeddings.name,
            "sha256": frozen.file_sha256(embeddings),
        },
    }
    manifest.write_text(json.dumps(record), encoding="utf-8")
    return {
        "dir": tmp_path,
        "manifest": manifest,
        "model": model,
        "checkpoint": checkpoint,
        "embeddings": embeddings,
    }


def test_sha256_matches_a_known_value(tmp_path):
    path = tmp_path / "file.txt"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert frozen.file_sha256(path) == expected


def test_verify_file_fails_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        frozen.verify_file(tmp_path / "missing.pt", "0" * 64)


def test_load_model_is_verified_frozen_and_in_eval_mode(frozen_files):
    model = frozen.load_frozen_model(
        manifest_path=frozen_files["manifest"], checkpoint_dir=frozen_files["dir"]
    )
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    for key, value in model.state_dict().items():
        assert torch.equal(value, frozen_files["model"].state_dict()[key])


def test_load_model_fails_when_checkpoint_changed(frozen_files):
    torch.save(WaferCNN().state_dict(), frozen_files["checkpoint"])
    with pytest.raises(ValueError, match="changed"):
        frozen.load_frozen_model(
            manifest_path=frozen_files["manifest"], checkpoint_dir=frozen_files["dir"]
        )


def test_load_embeddings_returns_the_split(frozen_files):
    x, y = frozen.load_frozen_embeddings(
        "val", frozen_files["manifest"], frozen_files["dir"]
    )
    assert x.shape == (2, 256) and y.tolist() == [0, 1]


def test_load_embeddings_fails_when_file_changed(frozen_files):
    np.savez(frozen_files["embeddings"], val_x=np.zeros((2, 256)), val_y=np.arange(2))
    with pytest.raises(ValueError, match="changed"):
        frozen.load_frozen_embeddings(
            "val", frozen_files["manifest"], frozen_files["dir"]
        )


def test_test_split_stays_closed(frozen_files):
    with pytest.raises(ValueError, match="test"):
        frozen.load_frozen_embeddings(
            "test", frozen_files["manifest"], frozen_files["dir"]
        )


def test_embed_and_predict_matches_the_model(wafer_batch):
    model = WaferCNN().eval()
    y = torch.arange(len(wafer_batch))
    loader = DataLoader(TensorDataset(wafer_batch, y), batch_size=3)
    h, pred, labels = frozen.embed_and_predict(model, loader, torch.device("cpu"))
    with torch.no_grad():
        assert np.allclose(h, model.embed(wafer_batch).numpy())
        assert np.array_equal(pred, model(wafer_batch).argmax(1).numpy())
    assert np.array_equal(labels, y.numpy())


def test_check_metrics_allows_rounding_only():
    recorded = dict.fromkeys(frozen.METRIC_KEYS, 0.9)
    frozen.check_metrics({**recorded, "f1_macro": 0.9005}, recorded)
    with pytest.raises(ValueError, match="f1_macro"):
        frozen.check_metrics({**recorded, "f1_macro": 0.89}, recorded)


def test_check_embeddings_ignores_scale_but_catches_changes():
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert frozen.check_embeddings(2 * x, x) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="differ"):
        frozen.check_embeddings(x[::-1], x)
    with pytest.raises(ValueError, match="Shape"):
        frozen.check_embeddings(x[:1], x)


def test_manifest_of_another_checkpoint_is_protected(frozen_files):
    manifest, same = (
        frozen_files["manifest"],
        frozen.file_sha256(frozen_files["checkpoint"]),
    )
    frozen.check_can_write(manifest, same, force=False)
    frozen.check_can_write(manifest, "other", force=True)
    with pytest.raises(FileExistsError):
        frozen.check_can_write(manifest, "other", force=False)


@pytest.mark.data
def test_real_frozen_model_reproduces_recorded_results():
    if not FROZEN_MANIFEST.exists():
        pytest.skip("frozen manifest not found")
    manifest = frozen.read_manifest()
    assert manifest["name"] == FROZEN_MODEL
    try:
        model = frozen.load_frozen_model()
        saved_x, saved_y = frozen.load_frozen_embeddings("val")
        loader = DataLoader(WaferDataset("val"), batch_size=512)
    except FileNotFoundError:
        pytest.skip("frozen files not found")
    x, pred, y = frozen.embed_and_predict(model, loader, torch.device("cpu"))
    assert np.array_equal(y, saved_y)
    frozen.check_embeddings(x, saved_x)
    frozen.check_metrics(evaluate(y, pred), manifest["val"])
