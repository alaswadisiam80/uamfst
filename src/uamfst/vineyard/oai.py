from __future__ import annotations

import csv

import torch

from uamfst.charter.vessels import DatasetSpec
from uamfst.pressing.bouquet import ClinicalTextTemplate, HashingTokenizer
from uamfst.vineyard.baskets import load_volume_slices
from uamfst.vineyard.harvest import ManifestDataset, Record, standardize


class OAIDataset(ManifestDataset):
    def __init__(
        self,
        root: str,
        grade_threshold: int = 2,
        slices_per_orientation: int = 3,
        image_size: int = 224,
        tokenizer: HashingTokenizer | None = None,
        seed: int = 0,
    ) -> None:
        spec = DatasetSpec(
            name="oai",
            num_classes=2,
            modalities=("image", "text"),
            slices_per_orientation=slices_per_orientation,
            image_size=image_size,
        )
        template = ClinicalTextTemplate(("age", "sex", "assessment", "laterality"))
        super().__init__(root, spec, template, tokenizer, seed)
        self.grade_threshold = grade_threshold

    def _records(self, split: str) -> list[Record]:
        index = self.root / f"oai_{split}.csv"
        if not index.exists():
            raise FileNotFoundError(f"missing OAI index {index}")
        records: list[Record] = []
        with index.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                grade = int(row["kl_grade"])
                records.append(
                    Record(
                        image_path=self.root / row["volume"],
                        label=int(grade >= self.grade_threshold),
                        metadata={
                            "age": row.get("age", "unknown"),
                            "sex": row.get("sex", "unknown"),
                            "assessment": f"KL grade {grade}",
                            "laterality": row.get("side", "knee"),
                        },
                    )
                )
        return records

    def _load_image(self, record: Record, train: bool) -> torch.Tensor:
        volume = load_volume_slices(
            record.image_path,
            self.spec.slices_per_orientation * 3,
            self.spec.image_size,
        )
        if train:
            volume = torch.stack([self.train_transform(slice_) for slice_ in volume], dim=0)
        return standardize(volume)
