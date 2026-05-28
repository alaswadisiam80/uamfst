from __future__ import annotations

import torch

from uamfst.fermentation.malolactic import (
    DirichletHead,
    expected_categorical_entropy,
    expected_probabilities,
)
from uamfst.fermentation.yeast import RandomFeatureGP


def test_sngp_shapes_and_variance_positive() -> None:
    layer = RandomFeatureGP(
        input_dim=32, hidden_dim=16, proj_dim=12, num_features=64, num_classes=3
    )
    hidden = torch.randn(5, 32)
    logits, projected, features = layer(hidden)
    assert logits.shape == (5, 3)
    assert projected.shape == (5, 12)
    assert features.shape == (5, 64)
    layer.accumulate_precision(features)
    variance = layer.variance(features)
    assert variance.shape == (5,)
    assert bool((variance > 0).all())


def test_sngp_distance_awareness() -> None:
    layer = RandomFeatureGP(input_dim=8, hidden_dim=8, proj_dim=8, num_features=256, num_classes=2)
    train = torch.randn(64, 8)
    _, _, train_features = layer(train)
    layer.accumulate_precision(train_features)
    _, _, near = layer(train[:1])
    _, _, far = layer(train[:1] + 25.0)
    assert float(layer.variance(far).detach()) > float(layer.variance(near).detach())


def test_dirichlet_head_valid_concentration() -> None:
    head = DirichletHead(proj_dim=10, hidden_dim=8, num_classes=4)
    concentration = head(torch.randn(6, 10))
    assert concentration.shape == (6, 4)
    assert bool((concentration > 1.0 - 1e-6).all())


def test_aleatoric_entropy_maximal_for_uniform() -> None:
    uniform = torch.ones(1, 5)
    peaked = torch.tensor([[50.0, 1.0, 1.0, 1.0, 1.0]])
    entropy_uniform = expected_categorical_entropy(uniform)
    entropy_peaked = expected_categorical_entropy(peaked)
    assert float(entropy_uniform) > float(entropy_peaked)
    assert torch.allclose(expected_probabilities(uniform), torch.full((1, 5), 0.2))
