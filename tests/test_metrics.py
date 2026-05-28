from __future__ import annotations

import torch
from sklearn.metrics import roc_auc_score

from uamfst.tasting.palate import (
    brier_score,
    expected_calibration_error,
    roc_auc,
    sensitivity_specificity,
)


def test_roc_auc_matches_sklearn() -> None:
    torch.manual_seed(1)
    labels = torch.randint(0, 2, (200,))
    scores = torch.rand(200) + labels.float() * 0.4
    reference = roc_auc_score(labels.numpy(), scores.numpy())
    assert abs(roc_auc(scores, labels) - float(reference)) < 1e-6


def test_perfect_separation_auc_is_one() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    assert abs(roc_auc(scores, labels) - 1.0) < 1e-9


def test_sensitivity_specificity_bounds() -> None:
    predictions = torch.tensor([1, 0, 1, 0])
    labels = torch.tensor([1, 0, 0, 0])
    sensitivity, specificity = sensitivity_specificity(predictions, labels)
    assert abs(sensitivity - 1.0) < 1e-9
    assert abs(specificity - (2.0 / 3.0)) < 1e-9


def test_brier_and_ece_for_confident_correct() -> None:
    probabilities = torch.tensor([[0.99, 0.01], [0.01, 0.99]])
    labels = torch.tensor([0, 1])
    assert brier_score(probabilities, labels) < 1e-3
    assert expected_calibration_error(probabilities, labels) < 0.05
