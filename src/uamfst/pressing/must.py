from __future__ import annotations

import math

import torch
from torch import nn


def _dct_basis(size: int) -> torch.Tensor:
    indices = torch.arange(size, dtype=torch.float64)
    freqs = torch.arange(size, dtype=torch.float64).unsqueeze(1)
    basis = torch.cos(math.pi / size * (indices + 0.5) * freqs)
    basis *= math.sqrt(2.0 / size)
    basis[0] *= 1.0 / math.sqrt(2.0)
    return basis.to(torch.float32)


def _semi_orthonormal(in_dim: int, out_dim: int, generator: torch.Generator) -> torch.Tensor:
    if out_dim >= in_dim:
        seed = torch.randn(out_dim, in_dim, generator=generator)
        q, _ = torch.linalg.qr(seed)
        oriented: torch.Tensor = q.transpose(0, 1).contiguous()
        return oriented
    seed = torch.randn(in_dim, out_dim, generator=generator)
    return seed / math.sqrt(in_dim)


class AnalyticBackbone(nn.Module):
    dct_basis: torch.Tensor
    vision_projection: torch.Tensor
    text_projection: torch.Tensor

    def __init__(
        self,
        vision_dim: int = 1024,
        text_dim: int = 4096,
        image_size: int = 224,
        dct_block: int = 16,
        text_hashes: int = 2048,
        seed: int = 20240115,
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.image_size = image_size
        self.dct_block = min(dct_block, image_size)
        self.text_hashes = text_hashes
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("dct_basis", _dct_basis(image_size))
        self.register_buffer(
            "vision_projection",
            _semi_orthonormal(self.dct_block * self.dct_block, vision_dim, generator),
        )
        self.register_buffer(
            "text_projection",
            _semi_orthonormal(text_hashes, text_dim, generator),
        )

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        gray = images.mean(dim=1)
        rolled = torch.einsum("ij,bjk->bik", self.dct_basis, gray)
        spectrum = torch.einsum("bik,lk->bil", rolled, self.dct_basis)
        block = spectrum[:, : self.dct_block, : self.dct_block]
        flat = torch.tanh(block.reshape(images.shape[0], -1))
        return flat @ self.vision_projection

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        rows = token_ids.shape[0]
        bag = torch.zeros((rows, self.text_hashes), dtype=torch.float32, device=token_ids.device)
        unigram = (token_ids * 2654435761 + 2246822519) % self.text_hashes
        bag.scatter_add_(1, unigram.long(), torch.ones_like(unigram, dtype=torch.float32))
        if token_ids.shape[1] > 1:
            pair = token_ids[:, :-1] * 1000003 + token_ids[:, 1:]
            bigram = (pair * 40503 + 2971215073) % self.text_hashes
            bag.scatter_add_(1, bigram.long(), torch.ones_like(bigram, dtype=torch.float32))
        bag = bag / bag.sum(dim=1, keepdim=True).clamp_min(1.0)
        projection = self.text_projection.to(bag.device)
        return bag @ projection
