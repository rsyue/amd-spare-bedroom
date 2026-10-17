"""VirtualGrid, VirtualDevice, and deterministic seeding for emulated ranks."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    """Deterministically seed Python, NumPy, and PyTorch (CPU and ROCm/CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@dataclass(frozen=True)
class VirtualDevice:
    """An emulated virtual GPU rank device within the grid."""

    rank: int
    memory_limit_bytes: int
    device_type: str
    address_space_id: str


class VirtualGrid:
    """Topology and resource manager for emulated virtual GPU ranks."""

    def __init__(
        self,
        world_size: int,
        total_memory_bytes: int,
        device_type: str,
        seed: int | None = None,
    ) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if total_memory_bytes <= 0:
            raise ValueError(f"total_memory_bytes must be > 0, got {total_memory_bytes}")

        self.world_size = world_size
        self.total_memory_bytes = total_memory_bytes
        self.per_rank_memory_bytes = total_memory_bytes // world_size
        self.device_type = device_type
        self.seed = seed
        self.ranks = list(range(world_size))

        if seed is not None:
            seed_everything(seed)

        base_seed = seed if seed is not None else 0
        self._generators: dict[int, torch.Generator] = {}
        for rank in self.ranks:
            gen = torch.Generator()
            gen.manual_seed(base_seed + rank)
            self._generators[rank] = gen

        self._devices: dict[int, VirtualDevice] = {
            rank: VirtualDevice(
                rank=rank,
                memory_limit_bytes=self.per_rank_memory_bytes,
                device_type=self.device_type,
                address_space_id=f"vdev:{rank}",
            )
            for rank in self.ranks
        }

    @classmethod
    def create(
        cls,
        world_size: int,
        total_memory_gb: float = 96.0,
        device_type: str = "auto",
        seed: int | None = None,
    ) -> VirtualGrid:
        """Create a new VirtualGrid partitioned across the requested world_size."""
        total_bytes = int(total_memory_gb * 1024 * 1024 * 1024)

        if device_type == "auto":
            try:
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device_type = "cpu"

        return cls(
            world_size=world_size,
            total_memory_bytes=total_bytes,
            device_type=device_type,
            seed=seed,
        )

    def get_device(self, rank: int) -> VirtualDevice:
        """Get the VirtualDevice corresponding to the given rank."""
        if rank < 0 or rank >= self.world_size:
            raise IndexError(f"Rank {rank} out of bounds for world_size {self.world_size}")
        return self._devices[rank]

    def get_generator(self, rank: int) -> torch.Generator:
        """Get the isolated RNG generator for a specific virtual rank."""
        if rank < 0 or rank >= self.world_size:
            raise IndexError(f"Rank {rank} out of bounds for world_size {self.world_size}")
        return self._generators[rank]
