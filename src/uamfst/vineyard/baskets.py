from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def _resize(image: torch.Tensor, size: int) -> torch.Tensor:
    resized = F.interpolate(
        image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0)


def load_rgb(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return _resize(tensor, size)


def load_volume_slices(path: Path, count: int, size: int) -> torch.Tensor:
    volume = np.load(path).astype(np.float32)
    depth = volume.shape[0]
    centre = depth // 2
    half = count // 2
    lo = max(0, centre - half)
    indices = list(range(lo, min(depth, lo + count)))
    while len(indices) < count:
        indices.append(indices[-1])
    span = max(float(volume.max()) - float(volume.min()), 1e-6)
    normalised = (volume - float(volume.min())) / span
    slices = []
    for index in indices:
        plane = torch.from_numpy(normalised[index]).unsqueeze(0).repeat(3, 1, 1)
        slices.append(_resize(plane, size))
    return torch.stack(slices, dim=0)
