from __future__ import annotations

import re
from collections.abc import Mapping

import torch


class ClinicalTextTemplate:
    def __init__(self, fields: tuple[str, ...]) -> None:
        self.fields = fields

    def render(self, metadata: Mapping[str, str]) -> str:
        parts = [f"{name}: {metadata[name]}" for name in self.fields if name in metadata]
        return " | ".join(parts)


class HashingTokenizer:
    def __init__(self, vocab_size: int = 32000, max_length: int = 64) -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length

    def encode(self, text: str) -> list[int]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        ids = [(hash_token(tok) % (self.vocab_size - 1)) + 1 for tok in tokens]
        ids = ids[: self.max_length]
        if not ids:
            ids = [0]
        return ids

    def batch_encode(self, texts: list[str]) -> torch.Tensor:
        encoded = [self.encode(text) for text in texts]
        width = max(len(seq) for seq in encoded)
        padded = [seq + [0] * (width - len(seq)) for seq in encoded]
        return torch.tensor(padded, dtype=torch.long)


def hash_token(token: str) -> int:
    value = 2166136261
    for char in token:
        value = (value ^ ord(char)) * 16777619 & 0xFFFFFFFF
    return value
