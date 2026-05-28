from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch

from uamfst.charter.vessels import Batch, DatasetSpec
from uamfst.pressing.bouquet import ClinicalTextTemplate, HashingTokenizer
from uamfst.vineyard.sorting import build_transform


@dataclass(frozen=True)
class Record:
    image_path: Path
    label: int
    metadata: dict[str, str] = field(default_factory=dict)


def standardize(image: torch.Tensor) -> torch.Tensor:
    return (image - 0.5) / 0.5


class ManifestDataset:
    def __init__(
        self,
        root: str,
        spec: DatasetSpec,
        template: ClinicalTextTemplate,
        tokenizer: HashingTokenizer | None = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root {self.root} does not exist")
        self.spec = spec
        self.template = template
        self.tokenizer = tokenizer or HashingTokenizer()
        self.train_transform = build_transform(train=True, seed=seed)
        self.eval_transform = build_transform(train=False, seed=seed)

    def _records(self, split: str) -> list[Record]:
        raise NotImplementedError

    def _load_image(self, record: Record, train: bool) -> torch.Tensor:
        raise NotImplementedError

    def _collate(self, records: list[Record], train: bool) -> Batch:
        images = torch.stack([self._load_image(record, train) for record in records], dim=0)
        texts = [self.template.render(record.metadata) for record in records]
        token_ids = self.tokenizer.batch_encode(texts)
        labels = torch.tensor([record.label for record in records], dtype=torch.long)
        return Batch(images=images, token_ids=token_ids, labels=labels)

    def batches(self, split: str, batch_size: int) -> Iterator[Batch]:
        records = self._records(split)
        train = split == "train"
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            yield self._collate(chunk, train)

    def support(self, shots: int, seed: int) -> Batch:
        records = self._records("train")
        generator = torch.Generator().manual_seed(seed)
        chosen: list[Record] = []
        for klass in range(self.spec.num_classes):
            members = [record for record in records if record.label == klass]
            order = torch.randperm(len(members), generator=generator)[:shots]
            chosen.extend(members[int(i)] for i in order)
        return self._collate(chosen, train=False)
