from __future__ import annotations

import os

import torch
from torch import distributed as dist


def init_distributed() -> bool:
    if not dist.is_available():
        return False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        return True
    return False


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
