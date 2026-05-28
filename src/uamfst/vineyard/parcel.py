from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from uamfst.charter.vessels import Batch, DatasetSpec


@runtime_checkable
class DataSource(Protocol):
    spec: DatasetSpec

    def batches(self, split: str, batch_size: int) -> Iterator[Batch]: ...

    def support(self, shots: int, seed: int) -> Batch: ...
