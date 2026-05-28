from __future__ import annotations

from collections.abc import Sequence

import torch
from absl import app, flags

from uamfst.appellation.controlled_origin import SplitConformal
from uamfst.appellation.declassification import DualCriterionDeferral
from uamfst.cellarmaster.station import configure
from uamfst.cooperage.logbook import get_logger
from uamfst.maturation.bottling import load_checkpoint

_CHECKPOINT = flags.DEFINE_string("checkpoint", "", "Optional checkpoint to restore before infer.")
_SPLIT = flags.DEFINE_string("split", "test", "Split to draw the demonstration batch from.")


def main(argv: Sequence[str]) -> None:
    del argv
    logger = get_logger()
    assembly = configure()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _CHECKPOINT.value:
        load_checkpoint(_CHECKPOINT.value, assembly.model)
    assembly.model.to(device).eval()

    calibration = next(assembly.data.batches("calibration", assembly.run.batch_size))
    batch = next(assembly.data.batches(_SPLIT.value, assembly.run.batch_size))
    with torch.no_grad():
        reference = assembly.model(calibration.to(device), passes=assembly.eval_passes)
        bundle = assembly.model(batch.to(device), passes=assembly.eval_passes)

    conformal: SplitConformal = assembly.conformal
    deferral: DualCriterionDeferral = assembly.deferral
    conformal.calibrate(
        reference.probabilities.cpu(),
        calibration.labels,
        reference.uncertainty.epistemic.cpu(),
        reference.uncertainty.aleatoric.cpu(),
    )
    deferral.calibrate(reference.uncertainty.epistemic.cpu())
    sets = conformal.prediction_sets(
        bundle.probabilities.cpu(),
        bundle.uncertainty.epistemic.cpu(),
        bundle.uncertainty.aleatoric.cpu(),
    )
    defer = deferral.should_defer(sets, bundle.uncertainty.epistemic.cpu())
    for i in range(batch.size):
        decision = "defer" if bool(defer[i]) else "diagnose"
        logger.info(
            "case %d | pred %d | p %.3f | epistemic %.4f | %s",
            i,
            int(bundle.probabilities[i].argmax()),
            float(bundle.probabilities[i].max()),
            float(bundle.uncertainty.epistemic[i]),
            decision,
        )


def entry() -> None:
    app.run(main)


if __name__ == "__main__":
    entry()
