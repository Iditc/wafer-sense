"""SimCLR: self-supervised wafer embeddings, learned without labels.

Two random views of each wafer pass through the same encoder and projection
head; the NT-Xent loss pulls the two views together and pushes the other
wafers in the batch away. After training only the encoder is kept: its
256-number output h is the embedding used by retrieval, open-set and probes.
The encoder has the same layers and names as WaferCNN.blocks, so its weights
can be loaded into the CNN for fine-tuning.
"""
import torch
import torch.nn.functional as F
from torch import nn

from src.models.cnn import conv_block

ADD_MAX = 0.01   # up to 1% of clean dies turn defective (lone failed dies)
DROP_MAX = 0.10  # up to 10% of defective dies turn clean


class SimCLR(nn.Module):
    """Encoder (4 conv blocks + average pool) and a 2-layer projection head."""

    def __init__(self, in_channels: int = 2, widths: tuple[int, ...] = (32, 64, 128, 256),
                 proj_dim: int = 128):
        super().__init__()
        chans = (in_channels, *widths)
        self.blocks = nn.Sequential(*[conv_block(a, b) for a, b in zip(chans, chans[1:])])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(widths[-1], widths[-1]), nn.ReLU(inplace=True),
                                  nn.Linear(widths[-1], proj_dim))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """h: the 256-number embedding kept after training."""
        return self.pool(self.blocks(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """z: the projection used only by the loss."""
        return self.head(self.embed(x))


def random_view(x: torch.Tensor, add_max: float = ADD_MAX, drop_max: float = DROP_MAX) -> torch.Tensor:
    """One random view per wafer of a 2-channel batch (runs on the GPU).

    Rotation by 0/90/180/270 degrees, a random horizontal flip, then die noise:
    each wafer draws its own add and drop rates, up to add_max and drop_max.
    No crop, so edge patterns stay whole.
    """
    out = x.clone()
    k = torch.randint(0, 4, (len(x),), device=x.device)
    for r in (1, 2, 3):
        idx = k == r
        out[idx] = torch.rot90(x[idx], r, dims=(2, 3))
    flip = torch.rand(len(x), device=x.device) < 0.5
    out[flip] = out[flip].flip(3)

    wafer, defect = out[:, 0], out[:, 1]
    p_add = torch.rand(len(x), 1, 1, device=x.device) * add_max
    p_drop = torch.rand(len(x), 1, 1, device=x.device) * drop_max
    r = torch.rand_like(defect)
    add = (wafer > 0) & (defect == 0) & (r < p_add)
    drop = (defect > 0) & (r < p_drop)
    defect = torch.where(add, 1.0, torch.where(drop, 0.0, defect))
    return torch.stack([wafer, defect], dim=1)


def nt_xent(za: torch.Tensor, zb: torch.Tensor, temperature: float) -> tuple[torch.Tensor, float]:
    """NT-Xent loss and the share of views whose most similar other view is their partner.

    For 2N views each view has 1 positive (its partner) and 2N-2 negatives.
    """
    z = F.normalize(torch.cat([za, zb]).float(), dim=1)
    n = len(za)
    sim = z @ z.T / temperature
    sim.fill_diagonal_(float("-inf"))  # a view is not compared with itself
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
    loss = F.cross_entropy(sim, target)
    acc = (sim.argmax(dim=1) == target).float().mean().item()
    return loss, acc
