"""Baseline CNN classifier for 2-channel wafer maps.

Four blocks of (Conv 3x3 -> BatchNorm -> ReLU) x 2 -> MaxPool 2x2.
Each block halves height and width and doubles the channels:
2x64x64 -> 32x32x32 -> 64x16x16 -> 128x8x8 -> 256x4x4 -> pool -> 9 classes.
1,175,241 parameters with the default widths.
"""
import torch
from torch import nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3x3 convs (size kept) then a 2x2 max pool (size halved).

    No conv bias: the BatchNorm right after it has its own shift.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class WaferCNN(nn.Module):
    """Conv blocks, global average pool, dropout and a linear class head."""

    def __init__(self, in_channels: int = 2, n_classes: int = 9,
                 widths: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        chans = (in_channels, *widths)
        self.blocks = nn.Sequential(*[conv_block(a, b) for a, b in zip(chans, chans[1:])])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(widths[-1], n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """256-number summary of each wafer (reused later for retrieval and open-set)."""
        return self.pool(self.blocks(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class scores (logits), one per class."""
        return self.head(self.dropout(self.embed(x)))


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
