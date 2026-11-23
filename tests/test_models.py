"""Unit tests: CNN, SimCLR, random views, NT-Xent loss and encoder loading."""

from argparse import Namespace

import pytest
import torch

import src.models.train_cnn as train_cnn
from src.models.cnn import WaferCNN, count_params
from src.models.simclr import SimCLR, nt_xent, random_view


def test_cnn_output_and_embedding_shapes(wafer_batch):
    model = WaferCNN().eval()
    assert model(wafer_batch).shape == (8, 9)
    assert model.embed(wafer_batch).shape == (8, 256)


def test_simclr_output_and_embedding_shapes(wafer_batch):
    model = SimCLR().eval()
    assert model(wafer_batch).shape == (8, 128)
    assert model.embed(wafer_batch).shape == (8, 256)


def test_parameter_counts_match_the_documented_architecture():
    assert count_params(WaferCNN()) == 1_175_241
    assert count_params(SimCLR()) == 1_271_616


def test_load_encoder_copies_blocks_and_keeps_head(tmp_path, monkeypatch):
    source = SimCLR()
    torch.save(source.state_dict(), tmp_path / "pretrained.pt")
    monkeypatch.setattr(train_cnn, "CHECKPOINT_DIR", tmp_path)
    model = WaferCNN()
    head_before = model.head.weight.clone()
    train_cnn.load_encoder(model, "pretrained")
    for name, value in model.blocks.state_dict().items():
        assert torch.equal(value, source.blocks.state_dict()[name])
    assert torch.equal(model.head.weight, head_before)


def test_load_encoder_fails_on_a_mismatched_checkpoint(tmp_path, monkeypatch):
    torch.save(SimCLR(widths=(16, 32, 64, 128)).state_dict(), tmp_path / "bad.pt")
    monkeypatch.setattr(train_cnn, "CHECKPOINT_DIR", tmp_path)
    with pytest.raises(RuntimeError):
        train_cnn.load_encoder(WaferCNN(), "bad")


def test_view_keeps_shape_and_binary_values(wafer_batch):
    view = random_view(wafer_batch)
    assert view.shape == wafer_batch.shape
    assert set(view.unique().tolist()) <= {0.0, 1.0}


def test_view_does_not_change_the_input(wafer_batch):
    before = wafer_batch.clone()
    random_view(wafer_batch)
    assert torch.equal(wafer_batch, before)


def test_view_without_noise_is_a_rotation_or_flip(wafer_batch):
    view = random_view(wafer_batch, add_max=0.0, drop_max=0.0)
    for original, out in zip(wafer_batch, view):
        rotated = [torch.rot90(original, k, dims=(1, 2)) for k in range(4)]
        options = rotated + [r.flip(2) for r in rotated]
        assert any(torch.equal(out, option) for option in options)


def test_view_noise_stays_inside_the_wafer(wafer_batch):
    view = random_view(wafer_batch, add_max=0.5, drop_max=0.5)
    assert (view[:, 1] <= view[:, 0]).all()


def test_view_noise_never_changes_the_wafer_area(wafer_batch):
    view = random_view(wafer_batch, add_max=0.5, drop_max=0.5)
    assert torch.equal(view[:, 0].sum((1, 2)), wafer_batch[:, 0].sum((1, 2)))


def test_view_drop_only_removes_defects(wafer_batch):
    view = random_view(wafer_batch, add_max=0.0, drop_max=0.5)
    assert (view[:, 1].sum((1, 2)) <= wafer_batch[:, 1].sum((1, 2))).all()


def test_view_add_only_adds_defects(wafer_batch):
    view = random_view(wafer_batch, add_max=0.5, drop_max=0.0)
    assert (view[:, 1].sum((1, 2)) >= wafer_batch[:, 1].sum((1, 2))).all()


def test_nt_xent_is_lower_for_matching_pairs():
    torch.manual_seed(0)
    z = torch.randn(32, 16)
    matched, matched_acc = nt_xent(z, z + 0.01 * torch.randn_like(z), 0.2)
    random, _ = nt_xent(z, torch.randn(32, 16), 0.2)
    assert matched < random
    assert matched_acc == 1.0


def test_nt_xent_accuracy_is_low_for_random_pairs():
    torch.manual_seed(0)
    _, acc = nt_xent(torch.randn(64, 16), torch.randn(64, 16), 0.2)
    assert acc < 0.2


def test_nt_xent_does_not_depend_on_vector_length():
    torch.manual_seed(0)
    a, b = torch.randn(16, 8), torch.randn(16, 8)
    assert nt_xent(a, b, 0.5)[0].item() == pytest.approx(
        nt_xent(5 * a, 2 * b, 0.5)[0].item()
    )


def test_nt_xent_gradients_are_finite():
    torch.manual_seed(0)
    z = torch.randn(8, 16, requires_grad=True)
    loss, _ = nt_xent(z, torch.randn(8, 16), 0.2)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()


def test_cnn_cli_accepts_the_fair_check_options(monkeypatch):
    argv = [
        "train_cnn",
        "--label-frac",
        "0.1",
        "--exclude",
        "Scratch",
        "--init",
        "simclr_t02",
    ]
    monkeypatch.setattr("sys.argv", argv)
    args: Namespace = train_cnn.parse_args()
    assert (args.label_frac, args.exclude, args.init) == (0.1, "Scratch", "simclr_t02")
