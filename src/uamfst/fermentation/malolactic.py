from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DirichletHead(nn.Module):
    def __init__(self, proj_dim: int = 256, hidden_dim: int = 128, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        evidence = F.softplus(self.net(projected))
        return evidence + 1.0


def expected_probabilities(concentration: torch.Tensor) -> torch.Tensor:
    return concentration / concentration.sum(dim=-1, keepdim=True)


def expected_categorical_entropy(concentration: torch.Tensor) -> torch.Tensor:
    mean = expected_probabilities(concentration)
    return -(mean * torch.log(mean.clamp_min(1e-12))).sum(dim=-1)
