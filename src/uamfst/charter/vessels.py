from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class Batch:
    images: torch.Tensor
    token_ids: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> Batch:
        return Batch(
            images=self.images.to(device),
            token_ids=self.token_ids.to(device),
            labels=self.labels.to(device),
        )

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True, eq=False)
class BackboneOutput:
    vision: torch.Tensor
    text: torch.Tensor


@dataclass(frozen=True, eq=False)
class UncertaintyReadout:
    epistemic: torch.Tensor
    aleatoric: torch.Tensor
    epistemic_vision: torch.Tensor
    epistemic_text: torch.Tensor


@dataclass(frozen=True, eq=False)
class PredictionBundle:
    logits: torch.Tensor
    probabilities: torch.Tensor
    concentration: torch.Tensor
    fused: torch.Tensor
    uncertainty: UncertaintyReadout


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    modalities: tuple[str, ...]
    slices_per_orientation: int = 3
    image_size: int = 224


@dataclass(frozen=True)
class RunConfig:
    seed: int
    epochs: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    precision: str
    world_size: int

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum * self.world_size
