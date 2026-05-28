from __future__ import annotations

from collections.abc import Iterator

import torch

from uamfst.charter.vessels import Batch, DatasetSpec


class SyntheticCohort:
    def __init__(
        self,
        num_classes: int = 2,
        image_size: int = 32,
        slices_per_orientation: int = 1,
        vocab_size: int = 1024,
        text_length: int = 16,
        train_size: int = 64,
        eval_size: int = 32,
        signal: float = 2.5,
        seed: int = 7,
    ) -> None:
        self.spec = DatasetSpec(
            name="synthetic",
            num_classes=num_classes,
            modalities=("image", "text"),
            slices_per_orientation=slices_per_orientation,
            image_size=image_size,
        )
        self.vocab_size = vocab_size
        self.text_length = text_length
        self.signal = signal
        self.slices = max(1, slices_per_orientation) * 3
        generator = torch.Generator().manual_seed(seed)
        self._image_prototype = torch.randn(
            num_classes, 3, image_size, image_size, generator=generator
        )
        self._token_bias = torch.randint(
            1, vocab_size, (num_classes, text_length), generator=generator
        )
        self._splits = {
            "train": self._materialise(train_size, generator),
            "val": self._materialise(eval_size, generator),
            "calibration": self._materialise(eval_size, generator),
            "test": self._materialise(eval_size, generator),
        }

    def _materialise(self, size: int, generator: torch.Generator) -> Batch:
        labels = torch.randint(0, self.spec.num_classes, (size,), generator=generator)
        noise = torch.randn(
            size, self.slices, 3, self.spec.image_size, self.spec.image_size, generator=generator
        )
        prototype = self._image_prototype[labels].unsqueeze(1)
        images = self.signal * prototype + noise
        tokens = torch.randint(1, self.vocab_size, (size, self.text_length), generator=generator)
        keep = torch.rand(size, self.text_length, generator=generator) < 0.5
        tokens = torch.where(keep, self._token_bias[labels], tokens)
        return Batch(images=images, token_ids=tokens, labels=labels)

    def batches(self, split: str, batch_size: int) -> Iterator[Batch]:
        pool = self._splits[split]
        for start in range(0, pool.size, batch_size):
            stop = min(start + batch_size, pool.size)
            yield Batch(
                images=pool.images[start:stop],
                token_ids=pool.token_ids[start:stop],
                labels=pool.labels[start:stop],
            )

    def support(self, shots: int, seed: int) -> Batch:
        pool = self._splits["train"]
        generator = torch.Generator().manual_seed(seed)
        chosen: list[int] = []
        for klass in range(self.spec.num_classes):
            members = (pool.labels == klass).nonzero(as_tuple=True)[0]
            order = torch.randperm(members.numel(), generator=generator)[:shots]
            chosen.extend(members[order].tolist())
        index = torch.tensor(chosen)
        return Batch(
            images=pool.images[index],
            token_ids=pool.token_ids[index],
            labels=pool.labels[index],
        )
