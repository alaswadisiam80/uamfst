from __future__ import annotations

import math
from pathlib import Path

import torch

from uamfst.cellarmaster.station import build_assembly
from uamfst.maturation.aging import TrainingLoop
from uamfst.maturation.inoculation import set_seed
from uamfst.tasting.judging import Evaluator

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_smoke_training_reduces_loss_and_reports(monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    assembly = build_assembly(["configs/experiment/_smoke.gin"])
    set_seed(assembly.run.seed)
    optimizer = torch.optim.AdamW(
        [p for p in assembly.model.parameters() if p.requires_grad], lr=1e-2
    )
    loop = TrainingLoop(assembly.model, assembly.objective, optimizer, train_passes=2)
    batches = list(assembly.data.batches("train", assembly.run.batch_size))
    history: list[float] = []
    for _ in range(8):
        history.extend(loop.run(batches))
    assert all(math.isfinite(value) for value in history)
    assert min(history[-3:]) < history[0]

    loop.calibrate_precision(assembly.data, assembly.run.batch_size)
    evaluator = Evaluator(
        assembly.conformal, assembly.deferral, assembly.run.batch_size, assembly.eval_passes
    )
    report = evaluator.evaluate(assembly.model, assembly.data)
    assert 0.0 <= report.coverage <= 1.0
    assert 0.0 <= report.conformal_coverage <= 1.0
    assert math.isfinite(report.ece)
