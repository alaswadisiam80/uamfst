from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _rotate(image: torch.Tensor, degrees: float) -> torch.Tensor:
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=image.dtype)
    grid = F.affine_grid(theta.unsqueeze(0), [1, *image.shape], align_corners=False)
    warped = F.grid_sample(image.unsqueeze(0), grid, align_corners=False)
    return warped.squeeze(0)


class ImageAugmentation:
    def __init__(
        self,
        train: bool,
        max_rotation: float = 15.0,
        jitter: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.train = train
        self.max_rotation = max_rotation
        self.jitter = jitter
        self._generator = torch.Generator().manual_seed(seed)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if not self.train:
            return image
        if torch.rand(1, generator=self._generator).item() < 0.5:
            image = torch.flip(image, dims=[-1])
        angle = (torch.rand(1, generator=self._generator).item() * 2 - 1) * self.max_rotation
        image = _rotate(image, angle)
        scale = 1.0 + (torch.rand(1, generator=self._generator).item() * 2 - 1) * self.jitter
        return (image * scale).clamp(image.min().item(), image.max().item())


def build_transform(train: bool, seed: int = 0) -> ImageAugmentation:
    return ImageAugmentation(train=train, seed=seed)
