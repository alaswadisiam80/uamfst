from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from absl import app

from uamfst.cellarmaster.station import configure
from uamfst.cooperage.casks import write_json
from uamfst.cooperage.logbook import get_logger
from uamfst.maturation.aging import TrainingLoop
from uamfst.maturation.bottling import save_checkpoint
from uamfst.maturation.inoculation import set_seed
from uamfst.maturation.racking import build_optimizer, build_schedule
from uamfst.tasting.judging import Evaluator


def main(argv: Sequence[str]) -> None:
    del argv
    logger = get_logger()
    assembly = configure()
    set_seed(assembly.run.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    optimizer = build_optimizer(
        assembly.model, assembly.run.learning_rate, assembly.run.weight_decay
    )
    steps_per_epoch = max(1, len(list(assembly.data.batches("train", assembly.run.batch_size))))
    total_steps = steps_per_epoch * assembly.run.epochs
    warmup_steps = steps_per_epoch * assembly.run.warmup_epochs
    scheduler = build_schedule(optimizer, warmup_steps, total_steps)

    loop = TrainingLoop(
        assembly.model,
        assembly.objective,
        optimizer,
        scheduler,
        device=device,
        grad_accum=assembly.run.grad_accum,
        train_passes=assembly.train_passes,
    )
    history = loop.fit(assembly.data, assembly.run)
    logger.info("trained %d steps, final loss %.4f", len(history), history[-1])

    evaluator = Evaluator(
        assembly.conformal,
        assembly.deferral,
        batch_size=assembly.run.batch_size,
        passes=assembly.eval_passes,
        device=device,
    )
    report = evaluator.evaluate(assembly.model, assembly.data)
    logger.info("AUC %.3f | ECE %.3f | coverage %.3f", report.auc, report.ece, report.coverage)

    output = Path(assembly.output_dir)
    write_json(output / "report.json", asdict(report))
    save_checkpoint(
        output / "model.ckpt",
        assembly.model,
        optimizer,
        assembly.run.epochs,
        assembly.run.seed,
    )


def entry() -> None:
    app.run(main)


if __name__ == "__main__":
    entry()
