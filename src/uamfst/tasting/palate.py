from __future__ import annotations

import numpy as np
import torch
from scipy.stats import rankdata


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    values = scores.detach().cpu().numpy().astype(np.float64)
    targets = labels.detach().cpu().numpy().astype(np.int64)
    positives = targets == 1
    negatives = targets == 0
    count_pos = int(positives.sum())
    count_neg = int(negatives.sum())
    if count_pos == 0 or count_neg == 0:
        return float("nan")
    ranks = rankdata(values)
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - count_pos * (count_pos + 1) / 2.0) / (count_pos * count_neg)


def sensitivity_specificity(predictions: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    pred = predictions.detach().cpu()
    target = labels.detach().cpu()
    true_positive = float(((pred == 1) & (target == 1)).sum())
    false_negative = float(((pred == 0) & (target == 1)).sum())
    true_negative = float(((pred == 0) & (target == 0)).sum())
    false_positive = float(((pred == 1) & (target == 0)).sum())
    sensitivity = true_positive / max(true_positive + false_negative, 1.0)
    specificity = true_negative / max(true_negative + false_positive, 1.0)
    return sensitivity, specificity


def expected_calibration_error(
    probabilities: torch.Tensor, labels: torch.Tensor, num_bins: int = 15
) -> float:
    confidence, prediction = probabilities.max(dim=-1)
    correct = (prediction == labels).double()
    confidence = confidence.double()
    edges = torch.linspace(0.0, 1.0, num_bins + 1)
    total = confidence.shape[0]
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
        error += gap * count / total
    return error


def brier_score(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    one_hot = torch.zeros_like(probabilities)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    return float((probabilities - one_hot).pow(2).sum(dim=-1).mean())
