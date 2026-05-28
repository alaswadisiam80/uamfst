from __future__ import annotations

import json

import torch

from uamfst.charter.vessels import DatasetSpec
from uamfst.pressing.bouquet import ClinicalTextTemplate, HashingTokenizer
from uamfst.vineyard.baskets import load_rgb
from uamfst.vineyard.harvest import ManifestDataset, Record, standardize


class VQADataset(ManifestDataset):
    def __init__(
        self,
        root: str,
        name: str = "vqa-rad",
        image_size: int = 224,
        tokenizer: HashingTokenizer | None = None,
        seed: int = 0,
    ) -> None:
        spec = DatasetSpec(
            name=name,
            num_classes=2,
            modalities=("image", "text"),
            slices_per_orientation=1,
            image_size=image_size,
        )
        template = ClinicalTextTemplate(("question",))
        super().__init__(root, spec, template, tokenizer, seed)

    def _records(self, split: str) -> list[Record]:
        index = self.root / f"{split}.json"
        if not index.exists():
            raise FileNotFoundError(f"missing VQA index {index}")
        with index.open() as handle:
            payload = json.load(handle)
        records: list[Record] = []
        for item in payload:
            answer = str(item["answer"]).strip().lower()
            label = 1 if answer in {"yes", "present", "abnormal"} else 0
            records.append(
                Record(
                    image_path=self.root / "images" / str(item["image"]),
                    label=label,
                    metadata={"question": str(item["question"])},
                )
            )
        return records

    def _load_image(self, record: Record, train: bool) -> torch.Tensor:
        image = load_rgb(record.image_path, self.spec.image_size)
        if train:
            image = self.train_transform(image)
        return standardize(image.unsqueeze(0))
