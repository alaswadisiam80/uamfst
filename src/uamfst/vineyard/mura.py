from __future__ import annotations

import csv

import torch

from uamfst.charter.vessels import DatasetSpec
from uamfst.pressing.bouquet import ClinicalTextTemplate, HashingTokenizer
from uamfst.vineyard.baskets import load_rgb
from uamfst.vineyard.harvest import ManifestDataset, Record, standardize

_SPLIT_FILE = {"train": "train_image_paths.csv", "val": "valid_image_paths.csv"}


def _region(path: str) -> str:
    for part in path.split("/"):
        if part.startswith("XR_"):
            return part[3:].lower()
    return "unknown"


class MuraDataset(ManifestDataset):
    def __init__(
        self,
        root: str,
        image_size: int = 224,
        tokenizer: HashingTokenizer | None = None,
        seed: int = 0,
    ) -> None:
        spec = DatasetSpec(
            name="mura",
            num_classes=2,
            modalities=("image", "text"),
            slices_per_orientation=1,
            image_size=image_size,
        )
        template = ClinicalTextTemplate(("body_part", "laterality", "indication"))
        super().__init__(root, spec, template, tokenizer, seed)

    def _records(self, split: str) -> list[Record]:
        index = self.root / _SPLIT_FILE[split]
        if not index.exists():
            raise FileNotFoundError(f"missing MURA index {index}")
        records: list[Record] = []
        with index.open(newline="") as handle:
            for (relative,) in csv.reader(handle):
                label = 1 if "positive" in relative else 0
                records.append(
                    Record(
                        image_path=self.root / relative,
                        label=label,
                        metadata={
                            "body_part": _region(relative),
                            "laterality": "upper extremity",
                            "indication": "radiograph review",
                        },
                    )
                )
        return records

    def _load_image(self, record: Record, train: bool) -> torch.Tensor:
        image = load_rgb(record.image_path, self.spec.image_size)
        if train:
            image = self.train_transform(image)
        return standardize(image.unsqueeze(0))
