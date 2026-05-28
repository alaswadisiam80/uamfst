from __future__ import annotations

import torch
from torch import nn

from uamfst.charter.vessels import Batch, RunConfig
from uamfst.cuverie.cuvee import UAMFST
from uamfst.fining.adjustments import CompositeObjective
from uamfst.maturation.lees import ExponentialMovingAverage
from uamfst.vineyard.parcel import DataSource


class TrainingLoop:
    def __init__(
        self,
        model: UAMFST,
        objective: CompositeObjective,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
        device: str = "cpu",
        grad_accum: int = 1,
        max_grad_norm: float = 1.0,
        train_passes: int = 2,
        ema: ExponentialMovingAverage | None = None,
    ) -> None:
        self.model = model.to(device)
        self.objective = objective
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.train_passes = train_passes
        self.ema = ema
        self._micro = 0

    def _trainable(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def step(self, batch: Batch) -> float:
        self.model.train()
        located = batch.to(self.device)
        bundle = self.model(located, passes=self.train_passes)
        loss = self.objective(bundle.logits, bundle.concentration, located.labels)
        (loss.total / self.grad_accum).backward()
        self._micro += 1
        if self._micro % self.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(self._trainable(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)
        return float(loss.total.item())

    def run(self, batches: list[Batch], max_steps: int | None = None) -> list[float]:
        history: list[float] = []
        for index, batch in enumerate(batches):
            if max_steps is not None and index >= max_steps:
                break
            history.append(self.step(batch))
        return history

    def fit(self, source: DataSource, config: RunConfig) -> list[float]:
        history: list[float] = []
        for _ in range(config.epochs):
            history.extend(self.run(list(source.batches("train", config.batch_size))))
        self.calibrate_precision(source, config.batch_size)
        return history

    def calibrate_precision(self, source: DataSource, batch_size: int) -> None:
        self.model.sngp.reset_precision()
        self.model.eval()
        with torch.no_grad():
            for batch in source.batches("train", batch_size):
                fused = self.model.fused_features(batch.to(self.device), passes=1)
                self.model.refresh_precision(fused)
