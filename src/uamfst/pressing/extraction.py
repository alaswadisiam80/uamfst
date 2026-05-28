from __future__ import annotations

import torch
from torch import nn


class VisualProjection(nn.Module):
    def __init__(self, vision_dim: int, model_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(vision_dim, model_dim)

    def forward(self, vision: torch.Tensor) -> torch.Tensor:
        projected: torch.Tensor = self.linear(vision)
        return projected
