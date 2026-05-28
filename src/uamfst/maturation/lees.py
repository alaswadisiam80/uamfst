from __future__ import annotations

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(parameter, alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    parameter.copy_(self.shadow[name])
