from __future__ import annotations

import math

import torch
from torch import nn


def build_optimizer(
    model: nn.Module,
    learning_rate: float = 2e-5,
    weight_decay: float = 1e-2,
) -> torch.optim.Optimizer:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)


def build_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
