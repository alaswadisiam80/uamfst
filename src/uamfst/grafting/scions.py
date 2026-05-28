from __future__ import annotations

import torch
from torch import nn

EPSILON = 1e-6


def build_prototypes(
    features: torch.Tensor,
    epistemic: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    weights = 1.0 / (epistemic + epsilon)
    weighted = features * weights.unsqueeze(-1)
    prototypes = features.new_zeros((num_classes, features.shape[-1]))
    for klass in range(num_classes):
        mask = labels == klass
        count = int(mask.sum().item())
        if count == 0:
            continue
        prototypes[klass] = weighted[mask].sum(dim=0) / count
    return prototypes


class PrototypeClassifier(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(features, prototypes).pow(2)
        return -distances / (2.0 * self.temperature.pow(2))
