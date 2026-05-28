from __future__ import annotations

import torch

from uamfst.maceration.marrying import UncertaintyFusionGate, precision_weight
from uamfst.maceration.tannins import EpistemicCombiner, mc_dropout_variance


def test_precision_weight_is_monotone_decreasing() -> None:
    low = precision_weight(torch.tensor(0.01))
    high = precision_weight(torch.tensor(10.0))
    assert float(low) > float(high)


def test_fusion_collapses_to_certain_modality() -> None:
    gate = UncertaintyFusionGate()
    vision = torch.ones(2, 4)
    text = torch.zeros(2, 4)
    certain = torch.tensor([1e-6, 1e-6])
    uncertain = torch.tensor([1e3, 1e3])
    fused = gate(vision, text, certain, uncertain)
    assert torch.allclose(fused, vision, atol=1e-2)


def test_fusion_is_convex_combination() -> None:
    gate = UncertaintyFusionGate()
    vision = torch.zeros(3, 5)
    text = torch.ones(3, 5)
    fused = gate(vision, text, torch.rand(3), torch.rand(3))
    assert bool((fused >= -1e-6).all()) and bool((fused <= 1.0 + 1e-6).all())


def test_mc_dropout_variance_zero_for_constant_samples() -> None:
    samples = torch.ones(7, 4, 3)
    assert float(mc_dropout_variance(samples).abs().max()) < 1e-8


def test_epistemic_combiner_between_inputs() -> None:
    combiner = EpistemicCombiner(init_gate=0.0)
    sngp = torch.tensor([1.0, 2.0])
    dropout = torch.tensor([3.0, 4.0])
    combined = combiner(sngp, dropout)
    lower = torch.minimum(sngp, dropout)
    upper = torch.maximum(sngp, dropout)
    assert bool((combined >= lower - 1e-6).all()) and bool((combined <= upper + 1e-6).all())
