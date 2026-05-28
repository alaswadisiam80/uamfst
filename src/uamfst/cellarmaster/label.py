from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from absl import app, flags
from torch import nn

from uamfst.cellarmaster.station import configure
from uamfst.cooperage.logbook import get_logger
from uamfst.cuverie.cuvee import UAMFST
from uamfst.maturation.bottling import load_checkpoint

_CHECKPOINT = flags.DEFINE_string("checkpoint", "", "Optional checkpoint to restore before export.")


class DecisionHead(nn.Module):
    def __init__(self, model: UAMFST) -> None:
        super().__init__()
        self.sngp = model.sngp
        self.dirichlet = model.dirichlet

    def forward(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, projected, _ = self.sngp(fused)
        return logits, self.dirichlet(projected)


def main(argv: Sequence[str]) -> None:
    del argv
    logger = get_logger()
    assembly = configure()
    if _CHECKPOINT.value:
        load_checkpoint(_CHECKPOINT.value, assembly.model)
    head = DecisionHead(assembly.model).eval()
    dummy = torch.zeros(1, assembly.model.model_dim)
    output = Path(assembly.output_dir) / "decision_head.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        head,
        (dummy,),
        str(output),
        input_names=["fused"],
        output_names=["logits", "concentration"],
        dynamic_axes={"fused": {0: "batch"}},
        opset_version=17,
    )
    logger.info("exported decision head to %s", output)


def entry() -> None:
    app.run(main)


if __name__ == "__main__":
    entry()
