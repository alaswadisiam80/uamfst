from __future__ import annotations

import torch

from uamfst.tasting.balance import interaction_ratio
from uamfst.tasting.palate import roc_auc
from uamfst.tasting.panel import bootstrap_ci, delong_test, holm_bonferroni, spearman_correlation


def test_delong_reports_difference_for_better_model() -> None:
    torch.manual_seed(2)
    labels = torch.randint(0, 2, (300,))
    strong = torch.rand(300) + labels.float() * 0.9
    weak = torch.rand(300) + labels.float() * 0.1
    auc_strong, auc_weak, _, pvalue = delong_test(strong, weak, labels)
    assert auc_strong > auc_weak
    assert pvalue < 0.05


def test_bootstrap_ci_brackets_point_estimate() -> None:
    torch.manual_seed(3)
    labels = torch.randint(0, 2, (400,))
    scores = torch.rand(400) + labels.float() * 0.5
    point = roc_auc(scores, labels)
    low, high = bootstrap_ci(roc_auc, scores, labels, resamples=200)
    assert low <= point <= high


def test_holm_bonferroni_rejects_small_pvalues() -> None:
    decisions = holm_bonferroni([0.001, 0.04, 0.8], alpha=0.05)
    assert decisions[0] is True
    assert decisions[2] is False


def test_spearman_perfect_monotone() -> None:
    first = torch.tensor([1.0, 2.0, 3.0, 4.0])
    second = torch.tensor([10.0, 20.0, 30.0, 40.0])
    assert abs(spearman_correlation(first, second) - 1.0) < 1e-9


def test_interaction_ratio_detects_synergy() -> None:
    ratio = interaction_ratio(
        full=0.96, without_first=0.963, without_second=0.931, without_both=0.923
    )
    assert ratio > 1.0
