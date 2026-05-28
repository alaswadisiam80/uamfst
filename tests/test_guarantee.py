from __future__ import annotations

import torch

from uamfst.appellation.controlled_origin import SplitConformal, nonconformity_scores
from uamfst.appellation.declassification import DualCriterionDeferral
from uamfst.tasting.flight import empirical_coverage


def test_split_conformal_achieves_nominal_coverage() -> None:
    torch.manual_seed(0)
    classes = 3
    total = 4000
    probabilities = torch.distributions.Dirichlet(torch.ones(classes)).sample((total,))
    labels = torch.randint(0, classes, (total,))
    epistemic = torch.rand(total)
    aleatoric = torch.rand(total)
    half = total // 2
    conformal = SplitConformal(alpha=0.1, uncertainty_weight=0.5)
    conformal.calibrate(probabilities[:half], labels[:half], epistemic[:half], aleatoric[:half])
    sets = conformal.prediction_sets(probabilities[half:], epistemic[half:], aleatoric[half:])
    coverage = empirical_coverage(sets, labels[half:])
    assert 0.86 <= coverage <= 0.97


def test_nonconformity_increases_with_uncertainty() -> None:
    probabilities = torch.tensor([[0.7, 0.3]])
    labels = torch.tensor([0])
    low = nonconformity_scores(probabilities, labels, torch.tensor([0.0]), torch.tensor([0.0]))
    high = nonconformity_scores(probabilities, labels, torch.tensor([1.0]), torch.tensor([1.0]))
    assert float(high) > float(low)


def test_dual_criterion_defers_high_epistemic_and_large_sets() -> None:
    epistemic = torch.linspace(0.0, 1.0, 100)
    deferral = DualCriterionDeferral(set_size_limit=1, epistemic_percentile=85.0)
    deferral.calibrate(epistemic)
    singleton = torch.zeros(100, 2, dtype=torch.bool)
    singleton[:, 0] = True
    defer = deferral.should_defer(singleton, epistemic)
    assert 0.10 <= float(defer.float().mean()) <= 0.20
    ambiguous = torch.ones(3, 2, dtype=torch.bool)
    forced = deferral.should_defer(ambiguous, torch.zeros(3))
    assert bool(forced.all())
