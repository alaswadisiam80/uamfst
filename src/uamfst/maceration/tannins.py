from __future__ import annotations

import torch
from torch import nn


def mc_dropout_variance(samples: torch.Tensor) -> torch.Tensor:
    centred = samples - samples.mean(dim=0, keepdim=True)
    per_dim = centred.pow(2).mean(dim=0)
    return per_dim.sum(dim=-1)


class EpistemicCombiner(nn.Module):
    def __init__(self, init_gate: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(init_gate))

    def forward(self, sngp_variance: torch.Tensor, dropout_variance: torch.Tensor) -> torch.Tensor:
        weight = torch.sigmoid(self.gate)
        return weight * sngp_variance + (1.0 - weight) * dropout_variance
