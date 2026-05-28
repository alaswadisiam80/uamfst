from __future__ import annotations

import torch
from torch import nn

from uamfst.charter.covenants import VisionLanguageBackbone


class TriOrientationPool(nn.Module):
    def __init__(self, backbone: VisionLanguageBackbone) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be an nn.Module implementing VisionLanguageBackbone")
        self.backbone = backbone
        self.vision_dim = backbone.vision_dim
        self.text_dim = backbone.text_dim

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 4:
            return self.backbone.encode_vision(images)
        batch, slices = images.shape[0], images.shape[1]
        merged = images.reshape(batch * slices, *images.shape[2:])
        features = self.backbone.encode_vision(merged)
        return features.reshape(batch, slices, -1).mean(dim=1)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_text(token_ids)
