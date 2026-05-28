from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


class TemperatureScaling(nn.Module):
    def __init__(self, init_temperature: float = 1.5) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self, logits: torch.Tensor, labels: torch.Tensor, steps: int = 200, lr: float = 0.05
    ) -> None:
        optimizer = torch.optim.Adam([self.log_temperature], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss: torch.Tensor = F.cross_entropy(self.forward(logits), labels)
            cast(Callable[[], None], loss.backward)()
            optimizer.step()
