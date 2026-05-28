from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class VisionLanguageBackbone(Protocol):
    vision_dim: int
    text_dim: int

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor: ...

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class UncertaintyModule(Protocol):
    def project(self, hidden: torch.Tensor) -> torch.Tensor: ...

    def epistemic(self, hidden: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class DeferralPolicy(Protocol):
    def should_defer(
        self,
        prediction_sets: torch.Tensor,
        epistemic: torch.Tensor,
    ) -> torch.Tensor: ...
