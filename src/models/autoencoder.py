"""Convolutional autoencoder for 2-channel wafer maps.

Encoder: the same 4 conv blocks as the CNN (each ends in a 2x2 max pool),
2x64x64 -> 256x4x4, then a linear layer to a small code (default 128 numbers).
Decoder: the mirror image. A linear layer back to 256x4x4, then 4 blocks that
each start with a nearest-neighbour upsample x2 and two 3x3 convs, back to 64x64.
Upsample + conv avoids the checkerboard marks of transposed convs, which an
anomaly score would mistake for defects.
2,825,858 parameters with the default sizes.
"""
import torch
from torch import nn

from src.models.cnn import conv_block


def up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Upsample x2 (size doubled), then two 3x3 convs (size kept)."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class WaferAE(nn.Module):
    """Encoder -> code -> decoder; forward returns logits of the rebuilt map."""

    def __init__(self, in_channels: int = 2, code_dim: int = 128, size: int = 64,
                 widths: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        chans = (in_channels, *widths)
        self.encoder = nn.Sequential(*[conv_block(a, b) for a, b in zip(chans, chans[1:])])
        self.inner = (widths[-1], size // 2 ** len(widths), size // 2 ** len(widths))  # 256x4x4
        flat = self.inner[0] * self.inner[1] * self.inner[2]
        self.to_code = nn.Linear(flat, code_dim)
        self.from_code = nn.Linear(code_dim, flat)
        back = (*widths[::-1], widths[0])  # 256 -> 128 -> 64 -> 32 -> 32
        self.decoder = nn.Sequential(*[up_block(a, b) for a, b in zip(back, back[1:])])
        self.out = nn.Conv2d(widths[0], in_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The code: a short summary of each wafer (reused later for retrieval)."""
        return self.to_code(self.encoder(x).flatten(1))

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        """Logits of the rebuilt map, same shape as the input."""
        x = self.from_code(code).view(-1, *self.inner)
        return self.out(self.decoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
