from __future__ import annotations

import torch


def coverage(defer_mask: torch.Tensor) -> float:
    return float((~defer_mask).double().mean())


def retained_accuracy(
    predictions: torch.Tensor, labels: torch.Tensor, defer_mask: torch.Tensor
) -> float:
    keep = ~defer_mask
    if int(keep.sum()) == 0:
        return float("nan")
    correct = (predictions[keep] == labels[keep]).double().mean()
    return float(correct)


def error_reduction(
    predictions: torch.Tensor, labels: torch.Tensor, defer_mask: torch.Tensor
) -> tuple[float, float, float]:
    full_error = float((predictions != labels).double().mean())
    keep = ~defer_mask
    if int(keep.sum()) == 0:
        return full_error, float("nan"), float("nan")
    retained_error = float((predictions[keep] != labels[keep]).double().mean())
    reduction = 1.0 - retained_error / full_error if full_error > 0 else 0.0
    return full_error, retained_error, reduction


def empirical_coverage(prediction_sets: torch.Tensor, labels: torch.Tensor) -> float:
    hit = prediction_sets.gather(1, labels.unsqueeze(1)).squeeze(1)
    return float(hit.double().mean())
