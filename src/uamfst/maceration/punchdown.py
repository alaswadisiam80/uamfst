from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def enable_mc_dropout(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Dropout | nn.Dropout1d | nn.Dropout2d | nn.Dropout3d):
            layer.train()


def stochastic_passes(
    forward: Callable[[], torch.Tensor],
    model: nn.Module,
    passes: int,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    enable_mc_dropout(model)
    samples = []
    with torch.no_grad():
        for _ in range(passes):
            samples.append(forward())
    if was_training:
        model.train()
    return torch.stack(samples, dim=0)
