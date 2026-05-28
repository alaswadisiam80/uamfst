from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class RandomFeatureGP(nn.Module):
    rff_weight: torch.Tensor
    rff_bias: torch.Tensor
    precision: torch.Tensor

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        proj_dim: int = 256,
        num_features: int = 1024,
        num_classes: int = 2,
        length_scale: float = 2.0,
        ridge: float = 1e-3,
        seed: int = 13,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.num_features = num_features
        self.ridge = ridge
        self.feature_scale = math.sqrt(2.0 / num_features)
        self.project = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.GELU(),
            spectral_norm(nn.Linear(hidden_dim, proj_dim)),
        )
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "rff_weight",
            torch.randn(num_features, proj_dim, generator=generator) / length_scale,
        )
        self.register_buffer(
            "rff_bias",
            torch.rand(num_features, generator=generator) * 2.0 * math.pi,
        )
        self.beta = nn.Linear(num_features, num_classes, bias=False)
        self.register_buffer("precision", ridge * torch.eye(num_features))

    def projection(self, hidden: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.project(hidden)
        return out

    def random_features(self, projected: torch.Tensor) -> torch.Tensor:
        angle = projected @ self.rff_weight.t() + self.rff_bias
        return self.feature_scale * torch.cos(angle)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        scores: torch.Tensor = self.beta(features)
        return scores

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.projection(hidden)
        features = self.random_features(projected)
        return self.logits(features), projected, features

    def accumulate_precision(self, features: torch.Tensor) -> None:
        with torch.no_grad():
            self.precision += features.t() @ features

    def reset_precision(self) -> None:
        with torch.no_grad():
            eye = torch.eye(self.num_features, device=self.precision.device)
            self.precision.copy_(self.ridge * eye)

    def variance(self, features: torch.Tensor) -> torch.Tensor:
        covariance = torch.linalg.inv(self.precision)
        return torch.einsum("bi,ij,bj->b", features, covariance, features)
