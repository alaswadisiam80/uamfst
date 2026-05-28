from __future__ import annotations

import math

import torch


def nonconformity_scores(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    epistemic: torch.Tensor,
    aleatoric: torch.Tensor,
    uncertainty_weight: float = 0.5,
) -> torch.Tensor:
    true_probability = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    return 1.0 - true_probability + uncertainty_weight * (epistemic + aleatoric)


def conformal_quantile(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    count = scores.numel()
    rank = math.ceil((1.0 - alpha) * (count + 1))
    if rank > count:
        return torch.tensor(float("inf"), device=scores.device, dtype=scores.dtype)
    ordered, _ = torch.sort(scores)
    return ordered[rank - 1]


class SplitConformal:
    def __init__(self, alpha: float = 0.10, uncertainty_weight: float = 0.5) -> None:
        self.alpha = alpha
        self.uncertainty_weight = uncertainty_weight
        self._quantile: torch.Tensor | None = None

    def calibrate(
        self,
        probabilities: torch.Tensor,
        labels: torch.Tensor,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor,
    ) -> None:
        scores = nonconformity_scores(
            probabilities, labels, epistemic, aleatoric, self.uncertainty_weight
        )
        self._quantile = conformal_quantile(scores, self.alpha)

    @property
    def quantile(self) -> torch.Tensor:
        if self._quantile is None:
            raise RuntimeError("SplitConformal must be calibrated before use.")
        return self._quantile

    def prediction_sets(
        self,
        probabilities: torch.Tensor,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor,
    ) -> torch.Tensor:
        shift = self.uncertainty_weight * (epistemic + aleatoric)
        candidate_scores = 1.0 - probabilities + shift.unsqueeze(1)
        return candidate_scores <= self.quantile
