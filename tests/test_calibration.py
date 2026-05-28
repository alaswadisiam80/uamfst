from __future__ import annotations

import torch

from uamfst.fining.adjustments import CompositeObjective, kl_dirichlet_to_uniform
from uamfst.fining.clarity import differentiable_ece


def test_differentiable_ece_lower_for_calibrated_predictions() -> None:
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    calibrated = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9], [0.85, 0.15], [0.2, 0.8], [0.95, 0.05], [0.05, 0.95]]
    )
    overconfident = torch.tensor(
        [[0.99, 0.01], [0.99, 0.01], [0.99, 0.01], [0.99, 0.01], [0.99, 0.01], [0.01, 0.99]]
    )
    assert float(differentiable_ece(calibrated, labels)) < float(
        differentiable_ece(overconfident, labels)
    )


def test_ece_is_differentiable() -> None:
    logits = torch.randn(8, 2, requires_grad=True)
    error = differentiable_ece(torch.softmax(logits, dim=-1), torch.randint(0, 2, (8,)))
    error.backward()
    assert logits.grad is not None


def test_kl_to_uniform_zero_at_uniform() -> None:
    uniform = torch.ones(4, 3)
    assert float(kl_dirichlet_to_uniform(uniform).abs().max()) < 1e-5
    peaked = torch.tensor([[10.0, 1.0, 1.0]])
    assert float(kl_dirichlet_to_uniform(peaked)) > 0.0


def test_composite_objective_components_present() -> None:
    objective = CompositeObjective()
    logits = torch.randn(6, 2)
    concentration = torch.rand(6, 2) + 1.0
    labels = torch.randint(0, 2, (6,))
    breakdown = objective(logits, concentration, labels)
    assert breakdown.total.requires_grad is False or breakdown.total.dim() == 0
    assert float(breakdown.cross_entropy) >= 0.0
    assert float(breakdown.calibration) >= 0.0
