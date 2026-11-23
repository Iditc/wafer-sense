"""Unit tests: class weights, label sampling, datasets and data loaders."""

import numpy as np
import pytest
import torch

from src.data.plot_utils import CLASSES
from src.models.dataset import WaferDataset, class_weights, sample_labels
from src.models.train_utils import make_loaders


def test_class_weights_follow_inverse_sqrt_count():
    labels = np.concatenate([np.full(100 if c == 0 else 25, c) for c in range(9)])
    weights = class_weights(labels)
    assert weights[1] / weights[0] == pytest.approx(2.0)
    assert weights.mean().item() == pytest.approx(1.0)


def test_absent_class_gets_zero_weight():
    labels = np.repeat(np.arange(8), 10)
    weights = class_weights(labels)
    assert weights[8] == 0
    assert weights[:8].mean().item() == pytest.approx(1.0)


def test_sample_labels_takes_share_of_each_class():
    y = np.concatenate([np.zeros(1000, int), np.ones(100, int)])
    rows = sample_labels(y, 0.1)
    assert (y[rows] == 0).sum() == 100 and (y[rows] == 1).sum() == 10


def test_sample_labels_keeps_every_class():
    y = np.concatenate([np.zeros(1000, int), np.ones(3, int)])
    assert set(y[sample_labels(y, 0.01)]) == {0, 1}


def test_sample_labels_has_no_duplicates():
    rows = sample_labels(np.repeat(np.arange(9), 50), 0.5)
    assert len(rows) == len(set(rows))


def test_sample_labels_is_repeatable_and_seed_dependent():
    y = np.repeat(np.arange(3), 100)
    assert np.array_equal(sample_labels(y, 0.1, seed=1), sample_labels(y, 0.1, seed=1))
    assert not np.array_equal(
        sample_labels(y, 0.1, seed=1), sample_labels(y, 0.1, seed=2)
    )


def test_smaller_fraction_needs_no_more_than_larger():
    y = np.repeat(np.arange(9), 100)
    assert len(sample_labels(y, 0.01)) <= len(sample_labels(y, 0.1)) < len(y)


def test_dataset_returns_encoded_wafer_and_label(tiny_cache):
    data = WaferDataset("train")
    x, y = data[15]
    assert len(data) == len(tiny_cache["train"])
    assert x.shape == (2, 64, 64) and isinstance(x, torch.Tensor)
    assert y == tiny_cache["train"][15]


def test_dataset_keep_selects_rows_and_labels(tiny_cache):
    data = WaferDataset("train").keep(np.array([0, 25, 89]))
    assert len(data) == 3
    assert data.labels.tolist() == [0, 2, 8]


def test_loader_label_fraction_matches_probe_sample(tiny_cache):
    loaders = make_loaders(num_workers=0, batch_size=4, label_frac=0.2)
    expected = tiny_cache["train"][sample_labels(tiny_cache["train"], 0.2)]
    assert np.array_equal(loaders["train"].dataset.labels, expected)
    assert len(loaders["val"].dataset) == len(tiny_cache["val"])


def test_loader_exclude_removes_type_from_train_and_val(tiny_cache):
    scratch = CLASSES.index("Scratch")
    loaders = make_loaders(num_workers=0, batch_size=4, exclude="Scratch")
    for split in ("train", "val"):
        kept = loaders[split].dataset.labels
        assert scratch not in kept
        assert len(kept) == (tiny_cache[split] != scratch).sum()


def test_loader_rejects_unknown_excluded_type(tiny_cache):
    with pytest.raises(ValueError):
        make_loaders(num_workers=0, exclude="NotADefect")
