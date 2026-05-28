from __future__ import annotations

import torch
from torch import nn

EPSILON = 1e-6


def precision_weight(epistemic: torch.Tensor, epsilon: float = EPSILON) -> torch.Tensor:
    return 1.0 / (epistemic + epsilon)


class UncertaintyFusionGate(nn.Module):
    def __init__(self, epsilon: float = EPSILON) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        vision_hidden: torch.Tensor,
        text_hidden: torch.Tensor,
        vision_epistemic: torch.Tensor,
        text_epistemic: torch.Tensor,
    ) -> torch.Tensor:
        weight_v = precision_weight(vision_epistemic, self.epsilon).unsqueeze(-1)
        weight_t = precision_weight(text_epistemic, self.epsilon).unsqueeze(-1)
        numerator = weight_v * vision_hidden + weight_t * text_hidden
        return numerator / (weight_v + weight_t)
