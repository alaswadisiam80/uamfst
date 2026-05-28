from __future__ import annotations

from dataclasses import dataclass

from uamfst.appellation.controlled_origin import SplitConformal
from uamfst.appellation.declassification import DualCriterionDeferral
from uamfst.charter.vessels import RunConfig
from uamfst.cuverie.cuvee import UAMFST
from uamfst.cuverie.manifold import entrypoint, register
from uamfst.fining.adjustments import CompositeObjective
from uamfst.pressing.must import AnalyticBackbone
from uamfst.pressing.varietals import BiomedClipBackbone, LlavaMedBackbone
from uamfst.vineyard.mrnet import MRNetDataset
from uamfst.vineyard.mura import MuraDataset
from uamfst.vineyard.nursery import SyntheticCohort
from uamfst.vineyard.oai import OAIDataset
from uamfst.vineyard.parcel import DataSource
from uamfst.vineyard.vqa import VQADataset

register(AnalyticBackbone, "AnalyticBackbone")
register(LlavaMedBackbone, "LlavaMedBackbone")
register(BiomedClipBackbone, "BiomedClipBackbone")
register(UAMFST, "UAMFST")
register(CompositeObjective, "CompositeObjective")
register(SyntheticCohort, "SyntheticCohort")
register(MRNetDataset, "MRNetDataset")
register(MuraDataset, "MuraDataset")
register(OAIDataset, "OAIDataset")
register(VQADataset, "VQADataset")
register(SplitConformal, "SplitConformal")
register(DualCriterionDeferral, "DualCriterionDeferral")
register(RunConfig, "RunConfig")


@dataclass
class Assembly:
    model: UAMFST
    objective: CompositeObjective
    data: DataSource
    run: RunConfig
    conformal: SplitConformal
    deferral: DualCriterionDeferral
    eval_passes: int
    train_passes: int
    output_dir: str


@entrypoint
def assemble(
    model: UAMFST,
    objective: CompositeObjective,
    data: DataSource,
    run: RunConfig,
    conformal: SplitConformal,
    deferral: DualCriterionDeferral,
    eval_passes: int = 10,
    train_passes: int = 2,
    output_dir: str = "runs/uamfst",
) -> Assembly:
    return Assembly(
        model=model,
        objective=objective,
        data=data,
        run=run,
        conformal=conformal,
        deferral=deferral,
        eval_passes=eval_passes,
        train_passes=train_passes,
        output_dir=output_dir,
    )
