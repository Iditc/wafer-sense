"""Integration tests: data, model, training loop and metrics working together."""

from argparse import Namespace

import numpy as np
import torch
from torch import nn

import src.models.train_utils as train_utils
from src.models import train_simclr
from src.models.cnn import WaferCNN
from src.models.dataset import class_weights
from src.models.metrics import evaluate
from src.models.simclr import SimCLR


def copy_params(model: nn.Module) -> list[torch.Tensor]:
    """Detached copy of all trainable weights."""
    return [p.detach().clone() for p in model.parameters()]


def changed(model: nn.Module, before: list[torch.Tensor]) -> bool:
    """True if every trainable weight moved."""
    return all(not torch.equal(p, b) for p, b in zip(model.parameters(), before))


def test_cnn_training_step_updates_all_weights(tiny_cache):
    train_utils.set_seed(0)
    loader = train_utils.make_loaders(batch_size=16, num_workers=0)["train"]
    model = WaferCNN()
    before = copy_params(model)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(loader.dataset.labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_utils.train_epoch(
        model, loader, loss_fn, optimizer, torch.device("cpu")
    )
    assert np.isfinite(loss)
    assert changed(model, before)


def test_fit_saves_best_weights_and_metrics_run(tiny_cache, tmp_path, monkeypatch):
    monkeypatch.setattr(train_utils, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(train_utils, "get_device", lambda: torch.device("cpu"))
    train_utils.set_seed(0)
    loaders = train_utils.make_loaders(batch_size=16, num_workers=0)
    model = WaferCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = train_utils.fit(
        model, loaders, nn.CrossEntropyLoss(), optimizer, "tiny", epochs=2
    )
    assert len(history) == 2 and (tmp_path / "tiny.pt").exists()
    y, probs = train_utils.predict(model, loaders["val"], torch.device("cpu"))
    assert probs.shape == (len(tiny_cache["val"]), 9)
    assert np.allclose(probs.sum(axis=1), 1)
    assert 0 <= evaluate(y, probs.argmax(axis=1))["f1_macro"] <= 1


def test_fit_stops_early_without_improvement(tiny_cache, tmp_path, monkeypatch):
    monkeypatch.setattr(train_utils, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(train_utils, "get_device", lambda: torch.device("cpu"))
    loaders = train_utils.make_loaders(batch_size=16, num_workers=0)
    model = WaferCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # weights never change
    history = train_utils.fit(
        model, loaders, nn.CrossEntropyLoss(), optimizer, "flat", epochs=5, patience=1
    )
    assert len(history) == 2


def test_simclr_training_step_updates_the_encoder(wafer_batch):
    torch.manual_seed(0)
    model = SimCLR()
    before = copy_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(enabled=False)
    args = Namespace(add_max=0.01, drop_max=0.1, temperature=0.2)
    loss, acc = train_simclr.train_epoch(
        model, [wafer_batch], optimizer, scaler, args, torch.device("cpu")
    )
    assert np.isfinite(loss) and 0 <= acc <= 1
    assert changed(model, before)
