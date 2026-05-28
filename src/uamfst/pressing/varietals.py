from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn


def _require(module_name: str, hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(
            f"{hint} requires the '{module_name}' package, which is not installed. "
            f"Install the optional 'backbones' extra to enable real weights."
        )


class LlavaMedBackbone(nn.Module):
    vision_dim: int
    text_dim: int

    def __init__(
        self,
        weights: str = "microsoft/llava-med-v1.5-mistral-7b",
        vision_dim: int = 1024,
        text_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.weights = weights
        _require("transformers", "LlavaMedBackbone")
        _require("peft", "LoRA adaptation of LlavaMedBackbone")
        if not Path(weights).exists() and "/" not in weights:
            raise RuntimeError(f"LLaVA-Med weights not found at {weights!r}.")

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "LlavaMedBackbone weights are not materialised in this environment; "
            "use AnalyticBackbone for development or load weights on the target GPU."
        )

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "LlavaMedBackbone weights are not materialised in this environment; "
            "use AnalyticBackbone for development or load weights on the target GPU."
        )


class BiomedClipBackbone(nn.Module):
    vision_dim: int
    text_dim: int

    def __init__(
        self,
        weights: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        vision_dim: int = 768,
        text_dim: int = 768,
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.weights = weights
        _require("open_clip", "BiomedClipBackbone")

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "BiomedClipBackbone weights are not materialised in this environment; "
            "use AnalyticBackbone for development or load weights on the target GPU."
        )

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "BiomedClipBackbone weights are not materialised in this environment; "
            "use AnalyticBackbone for development or load weights on the target GPU."
        )
