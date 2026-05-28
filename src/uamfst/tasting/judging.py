from __future__ import annotations

from dataclasses import dataclass

import torch

from uamfst.appellation.controlled_origin import SplitConformal
from uamfst.appellation.declassification import DualCriterionDeferral
from uamfst.cuverie.cuvee import UAMFST
from uamfst.tasting.flight import coverage, empirical_coverage, error_reduction, retained_accuracy
from uamfst.tasting.palate import (
    brier_score,
    expected_calibration_error,
    roc_auc,
    sensitivity_specificity,
)
from uamfst.vineyard.parcel import DataSource


@dataclass(frozen=True, eq=False)
class Collected:
    probabilities: torch.Tensor
    labels: torch.Tensor
    epistemic: torch.Tensor
    aleatoric: torch.Tensor


@dataclass(frozen=True)
class EvaluationReport:
    auc: float
    ece: float
    brier: float
    sensitivity: float
    specificity: float
    coverage: float
    retained_accuracy: float
    error_reduction: float
    conformal_coverage: float


def collect(
    model: UAMFST,
    source: DataSource,
    split: str,
    batch_size: int,
    passes: int,
    device: str = "cpu",
) -> Collected:
    located = torch.device(device)
    model.eval()
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    epistemic: list[torch.Tensor] = []
    aleatoric: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in source.batches(split, batch_size):
            bundle = model(batch.to(located), passes=passes)
            probabilities.append(bundle.probabilities.cpu())
            labels.append(batch.labels.cpu())
            epistemic.append(bundle.uncertainty.epistemic.cpu())
            aleatoric.append(bundle.uncertainty.aleatoric.cpu())
    return Collected(
        torch.cat(probabilities),
        torch.cat(labels),
        torch.cat(epistemic),
        torch.cat(aleatoric),
    )


class Evaluator:
    def __init__(
        self,
        conformal: SplitConformal,
        deferral: DualCriterionDeferral,
        batch_size: int = 16,
        passes: int = 10,
        device: str = "cpu",
    ) -> None:
        self.conformal = conformal
        self.deferral = deferral
        self.batch_size = batch_size
        self.passes = passes
        self.device = device

    def evaluate(self, model: UAMFST, source: DataSource) -> EvaluationReport:
        calibration = collect(
            model, source, "calibration", self.batch_size, self.passes, self.device
        )
        test = collect(model, source, "test", self.batch_size, self.passes, self.device)
        self.conformal.calibrate(
            calibration.probabilities,
            calibration.labels,
            calibration.epistemic,
            calibration.aleatoric,
        )
        self.deferral.calibrate(calibration.epistemic)
        prediction_sets = self.conformal.prediction_sets(
            test.probabilities, test.epistemic, test.aleatoric
        )
        defer = self.deferral.should_defer(prediction_sets, test.epistemic)
        predictions = test.probabilities.argmax(dim=-1)
        sensitivity, specificity = sensitivity_specificity(predictions, test.labels)
        _, _, reduction = error_reduction(predictions, test.labels, defer)
        return EvaluationReport(
            auc=roc_auc(test.probabilities[:, 1], test.labels),
            ece=expected_calibration_error(test.probabilities, test.labels),
            brier=brier_score(test.probabilities, test.labels),
            sensitivity=sensitivity,
            specificity=specificity,
            coverage=coverage(defer),
            retained_accuracy=retained_accuracy(predictions, test.labels, defer),
            error_reduction=reduction,
            conformal_coverage=empirical_coverage(prediction_sets, test.labels),
        )
