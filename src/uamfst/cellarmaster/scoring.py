from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from absl import app, flags

from uamfst.cellarmaster.station import configure
from uamfst.cooperage.casks import write_json
from uamfst.cooperage.logbook import get_logger
from uamfst.maturation.bottling import load_checkpoint
from uamfst.tasting.judging import Evaluator

_CHECKPOINT = flags.DEFINE_string("checkpoint", "", "Optional checkpoint to restore before eval.")


def main(argv: Sequence[str]) -> None:
    del argv
    logger = get_logger()
    assembly = configure()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _CHECKPOINT.value:
        load_checkpoint(_CHECKPOINT.value, assembly.model)
    evaluator = Evaluator(
        assembly.conformal,
        assembly.deferral,
        batch_size=assembly.run.batch_size,
        passes=assembly.eval_passes,
        device=device,
    )
    report = evaluator.evaluate(assembly.model, assembly.data)
    logger.info(
        "AUC %.3f | ECE %.3f | Brier %.3f | coverage %.3f | retained acc %.3f",
        report.auc,
        report.ece,
        report.brier,
        report.coverage,
        report.retained_accuracy,
    )
    write_json(Path(assembly.output_dir) / "evaluation.json", asdict(report))


def entry() -> None:
    app.run(main)


if __name__ == "__main__":
    entry()
