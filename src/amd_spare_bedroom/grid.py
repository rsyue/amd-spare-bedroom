"""VirtualGrid and VirtualDevice topology for emulating distributed ranks."""

from __future__ import annotations

from dataclasses import dataclass


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
    ) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if total_memory_bytes <= 0:
            raise ValueError(f"total_memory_bytes must be > 0, got {total_memory_bytes}")

        self.world_size = world_size
        self.total_memory_bytes = total_memory_bytes
        self.per_rank_memory_bytes = total_memory_bytes // world_size
        self.device_type = device_type
        self.ranks = list(range(world_size))

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
    ) -> VirtualGrid:
        """Create a new VirtualGrid partitioned across the requested world_size."""
        total_bytes = int(total_memory_gb * 1024 * 1024 * 1024)

        if device_type == "auto":
            try:
                import torch

                device_type = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device_type = "cpu"

        return cls(
            world_size=world_size,
            total_memory_bytes=total_bytes,
            device_type=device_type,
        )

    def get_device(self, rank: int) -> VirtualDevice:
        """Get the VirtualDevice corresponding to the given rank."""
        if rank < 0 or rank >= self.world_size:
            raise IndexError(f"Rank {rank} out of bounds for world_size {self.world_size}")
        return self._devices[rank]
