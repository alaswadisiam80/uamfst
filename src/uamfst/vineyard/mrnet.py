from __future__ import annotations

import csv

import torch

from uamfst.charter.vessels import DatasetSpec
from uamfst.pressing.bouquet import ClinicalTextTemplate, HashingTokenizer
from uamfst.vineyard.baskets import load_volume_slices
from uamfst.vineyard.harvest import ManifestDataset, Record, standardize

_PLANES = ("axial", "coronal", "sagittal")
_SPLIT_DIR = {"train": "train", "val": "valid", "test": "valid"}


class MRNetDataset(ManifestDataset):
    def __init__(
        self,
        root: str,
        task: str = "abnormal",
        slices_per_orientation: int = 3,
        image_size: int = 224,
        tokenizer: HashingTokenizer | None = None,
        seed: int = 0,
    ) -> None:
        spec = DatasetSpec(
            name=f"mrnet-{task}",
            num_classes=2,
            modalities=("image", "text"),
            slices_per_orientation=slices_per_orientation,
            image_size=image_size,
        )
        template = ClinicalTextTemplate(("age", "sex", "indication", "laterality"))
        super().__init__(root, spec, template, tokenizer, seed)
        self.task = task

    def _records(self, split: str) -> list[Record]:
        directory = _SPLIT_DIR[split]
        label_file = self.root / f"{directory}-{self.task}.csv"
        if not label_file.exists():
            raise FileNotFoundError(f"missing MRNet label file {label_file}")
        records: list[Record] = []
        with label_file.open(newline="") as handle:
            for case, label in csv.reader(handle):
                records.append(
                    Record(
                        image_path=self.root / directory,
                        label=int(label),
                        metadata={"laterality": "knee", "indication": "internal derangement"},
                    )
                )
                records[-1].metadata["case"] = case
        return records

    def _load_image(self, record: Record, train: bool) -> torch.Tensor:
        case = record.metadata["case"]
        planes = [
            load_volume_slices(
                record.image_path / plane / f"{case}.npy",
                self.spec.slices_per_orientation,
                self.spec.image_size,
            )
            for plane in _PLANES
        ]
        volume = torch.cat(planes, dim=0)
        if train:
            volume = torch.stack([self.train_transform(slice_) for slice_ in volume], dim=0)
        return standardize(volume)
