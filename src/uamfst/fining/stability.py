from __future__ import annotations

import torch


def reliability_bins(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    confidence, prediction = probabilities.max(dim=-1)
    correct = (prediction == labels).to(probabilities.dtype)
    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=probabilities.device)
    bin_index = torch.bucketize(confidence, edges[1:-1])
    counts = torch.zeros(num_bins, device=probabilities.device)
    conf_sum = torch.zeros(num_bins, device=probabilities.device)
    acc_sum = torch.zeros(num_bins, device=probabilities.device)
    counts.scatter_add_(0, bin_index, torch.ones_like(confidence))
    conf_sum.scatter_add_(0, bin_index, confidence)
    acc_sum.scatter_add_(0, bin_index, correct)
    safe = counts.clamp_min(1.0)
    return conf_sum / safe, acc_sum / safe, counts
