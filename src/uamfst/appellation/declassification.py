from __future__ import annotations

import torch


class DualCriterionDeferral:
    def __init__(self, set_size_limit: int = 1, epistemic_percentile: float = 85.0) -> None:
        self.set_size_limit = set_size_limit
        self.epistemic_percentile = epistemic_percentile
        self._threshold: torch.Tensor | None = None

    def calibrate(self, epistemic: torch.Tensor) -> None:
        quantile = self.epistemic_percentile / 100.0
        self._threshold = torch.quantile(epistemic, quantile)

    @property
    def threshold(self) -> torch.Tensor:
        if self._threshold is None:
            raise RuntimeError("DualCriterionDeferral must be calibrated before use.")
        return self._threshold

    def should_defer(
        self,
        prediction_sets: torch.Tensor,
        epistemic: torch.Tensor,
    ) -> torch.Tensor:
        ambiguous = prediction_sets.sum(dim=-1) > self.set_size_limit
        uncertain = epistemic > self.threshold
        return ambiguous | uncertain
