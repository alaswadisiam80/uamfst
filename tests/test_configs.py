from __future__ import annotations

from pathlib import Path

import pytest

from uamfst.cuverie import assemblage
from uamfst.cuverie.manifold import clear, ingest

_ = assemblage

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = sorted((REPO_ROOT / "configs" / "experiment").glob("*.gin"))


@pytest.mark.parametrize("config", EXPERIMENTS, ids=lambda path: path.name)
def test_experiment_config_parses(config: Path, monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    clear()
    ingest([str(config.relative_to(REPO_ROOT))])


def test_every_reported_experiment_has_a_config() -> None:
    names = {path.name for path in EXPERIMENTS}
    required = {
        "main.gin",
        "main_mura.gin",
        "supplementary_biomedclip.gin",
        "supplementary_vqa.gin",
        "supplementary_oai_transfer.gin",
        "ablation_no_uq.gin",
        "ablation_no_fusion.gin",
        "ablation_no_adapter.gin",
        "ablation_uq_fusion.gin",
        "ablation_uq_adapter.gin",
        "ablation_fusion_adapter.gin",
        "ablation_base.gin",
        "ablation_temperature.gin",
        "_smoke.gin",
    }
    assert required <= names
