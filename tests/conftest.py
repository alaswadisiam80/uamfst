from __future__ import annotations

import pytest
import torch

from uamfst.charter.vessels import Batch
from uamfst.cuverie.cuvee import UAMFST
from uamfst.pressing.must import AnalyticBackbone
from uamfst.vineyard.nursery import SyntheticCohort


def make_model(num_classes: int = 2, dropout: float = 0.1, **kwargs: object) -> UAMFST:
    backbone = AnalyticBackbone(
        vision_dim=48, text_dim=96, image_size=16, dct_block=6, text_hashes=128
    )
    return UAMFST(
        backbone,
        num_classes=num_classes,
        sngp_hidden=48,
        sngp_proj=24,
        sngp_features=96,
        dirichlet_hidden=24,
        adapter_rank=8,
        dropout=dropout,
        mc_passes=2,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture
def cohort() -> SyntheticCohort:
    return SyntheticCohort(
        num_classes=2,
        image_size=16,
        slices_per_orientation=1,
        text_length=12,
        train_size=16,
        eval_size=8,
        signal=3.0,
        seed=11,
    )


@pytest.fixture
def batch(cohort: SyntheticCohort) -> Batch:
    return next(cohort.batches("train", 8))


@pytest.fixture
def model() -> UAMFST:
    torch.manual_seed(0)
    return make_model()
