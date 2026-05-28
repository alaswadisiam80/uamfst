from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from uamfst.fining.clarity import differentiable_ece


def kl_dirichlet_to_uniform(concentration: torch.Tensor) -> torch.Tensor:
    uniform = torch.ones_like(concentration)
    conc_sum = concentration.sum(dim=-1)
    uniform_sum = uniform.sum(dim=-1)
    digamma_diff = torch.digamma(concentration) - torch.digamma(conc_sum).unsqueeze(-1)
    return (
        torch.lgamma(conc_sum)
        - torch.lgamma(concentration).sum(dim=-1)
        - torch.lgamma(uniform_sum)
        + torch.lgamma(uniform).sum(dim=-1)
        + ((concentration - uniform) * digamma_diff).sum(dim=-1)
    )


@dataclass(frozen=True, eq=False)
class LossBreakdown:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    calibration: torch.Tensor
    dirichlet: torch.Tensor


class CompositeObjective(nn.Module):
    def __init__(
        self,
        calibration_weight: float = 0.1,
        dirichlet_weight: float = 0.01,
        num_bins: int = 15,
    ) -> None:
        super().__init__()
        self.calibration_weight = calibration_weight
        self.dirichlet_weight = dirichlet_weight
        self.num_bins = num_bins

    def forward(
        self,
        logits: torch.Tensor,
        concentration: torch.Tensor,
        labels: torch.Tensor,
    ) -> LossBreakdown:
        cross_entropy = F.cross_entropy(logits, labels)
        probabilities = torch.softmax(logits, dim=-1)
        calibration = differentiable_ece(probabilities, labels, self.num_bins)
        dirichlet = kl_dirichlet_to_uniform(concentration).mean()
        total = (
            cross_entropy
            + self.calibration_weight * calibration
            + self.dirichlet_weight * dirichlet
        )
        return LossBreakdown(total, cross_entropy, calibration, dirichlet)
