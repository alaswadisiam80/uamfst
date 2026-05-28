from __future__ import annotations

import torch
from torch import nn


class LowRankAdapter(nn.Module):
    def __init__(self, model_dim: int, rank: int = 16, scale: float = 1.0) -> None:
        super().__init__()
        self.down = nn.Linear(model_dim, rank, bias=False)
        self.up = nn.Linear(rank, model_dim, bias=False)
        self.scale = scale / rank
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = self.up(self.down(hidden))
        return hidden + self.scale * residual
