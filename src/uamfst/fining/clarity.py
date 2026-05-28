from __future__ import annotations

import torch


def differentiable_ece(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
    bandwidth: float = 0.05,
) -> torch.Tensor:
    confidence, prediction = probabilities.max(dim=-1)
    correct = (prediction == labels).to(probabilities.dtype)
    centres = torch.linspace(
        1.0 / (2 * num_bins),
        1.0 - 1.0 / (2 * num_bins),
        num_bins,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    distance = confidence.unsqueeze(1) - centres.unsqueeze(0)
    affinity = torch.softmax(-distance.pow(2) / (2.0 * bandwidth**2), dim=1)
    mass = affinity.sum(dim=0).clamp_min(1e-6)
    bin_confidence = (affinity * confidence.unsqueeze(1)).sum(dim=0) / mass
    bin_accuracy = (affinity * correct.unsqueeze(1)).sum(dim=0) / mass
    weight = mass / confidence.shape[0]
    return (weight * (bin_accuracy - bin_confidence).abs()).sum()
