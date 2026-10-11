"""VirtualCommunicator and collective communication primitives across virtual ranks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from amd_spare_bedroom.grid import VirtualGrid

ReductionOp = Literal["sum", "mean"]


class VirtualCommunicator:
    """Collective communication engine operating across emulated virtual GPU ranks."""

    def __init__(self, grid: VirtualGrid) -> None:
        self.grid = grid
        self.barrier_count: int = 0

    def barrier(self) -> None:
        """Execute a lockstep synchronization barrier across all ranks in the grid."""
        self.barrier_count += 1

    def all_reduce(
        self,
        rank_tensors: dict[int, torch.Tensor],
        op: ReductionOp = "mean",
    ) -> dict[int, torch.Tensor]:
        """Perform an All-Reduce operation across all virtual ranks.

        Args:
            rank_tensors: Mapping from rank ID to the tensor contributed by that rank.
            op: Reduction operation ('sum' or 'mean').

        Returns:
            Mapping from rank ID to the reduced tensor copy.
        """
        expected_ranks = set(self.grid.ranks)
        actual_ranks = set(rank_tensors.keys())

        if actual_ranks != expected_ranks:
            missing = expected_ranks - actual_ranks
            extra = actual_ranks - expected_ranks
            raise ValueError(
                f"Tensors must be provided for all ranks. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )

        # Validate shapes and dtypes match across all ranks
        first_tensor = rank_tensors[self.grid.ranks[0]]
        for rank in self.grid.ranks[1:]:
            t = rank_tensors[rank]
            if t.shape != first_tensor.shape:
                raise ValueError(
                    f"Shape mismatch at rank {rank}: expected {first_tensor.shape}, got {t.shape}"
                )
            if t.dtype != first_tensor.dtype:
                raise ValueError(
                    f"Dtype mismatch at rank {rank}: expected {first_tensor.dtype}, got {t.dtype}"
                )

        # Perform reduction
        accumulated = torch.zeros_like(first_tensor)
        for rank in self.grid.ranks:
            accumulated = accumulated + rank_tensors[rank]

        if op == "mean":
            reduced = accumulated / float(self.grid.world_size)
        elif op == "sum":
            reduced = accumulated
        else:
            raise ValueError(f"Unsupported reduction op: '{op}', expected 'sum' or 'mean'")

        # Distribute a fresh tensor copy to each rank
        self.barrier()
        return {rank: reduced.clone() for rank in self.grid.ranks}

    def broadcast(
        self,
        src_rank: int,
        tensor: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Broadcast a tensor from src_rank to all ranks in the grid.

        Args:
            src_rank: Source rank originating the tensor.
            tensor: Tensor to broadcast.

        Returns:
            Mapping from rank ID to the broadcasted tensor copy.
        """
        if src_rank not in self.grid.ranks:
            raise IndexError(
                f"Source rank {src_rank} out of bounds for world_size {self.grid.world_size}"
            )

        self.barrier()
        return {rank: tensor.clone() for rank in self.grid.ranks}

    def all_gather(
        self,
        rank_tensors: dict[int, torch.Tensor],
        dim: int = 0,
    ) -> dict[int, torch.Tensor]:
        """Gather tensor shards from all ranks and concatenate them along dim.

        Args:
            rank_tensors: Mapping from rank ID to that rank's tensor shard.
            dim: Dimension along which to concatenate gathered shards.

        Returns:
            Mapping from rank ID to the concatenated full tensor.
        """
        expected_ranks = set(self.grid.ranks)
        actual_ranks = set(rank_tensors.keys())

        if actual_ranks != expected_ranks:
            missing = expected_ranks - actual_ranks
            extra = actual_ranks - expected_ranks
            raise ValueError(
                f"Tensors must be provided for all ranks. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )

        # Concatenate shards in canonical rank order
        ordered_shards = [rank_tensors[rank] for rank in self.grid.ranks]
        gathered = torch.cat(ordered_shards, dim=dim)

        self.barrier()
        return {rank: gathered.clone() for rank in self.grid.ranks}

    def reduce_scatter(
        self,
        rank_tensors: dict[int, torch.Tensor],
        op: ReductionOp = "mean",
        dim: int = 0,
    ) -> dict[int, torch.Tensor]:
        """Reduce full tensors across all ranks and scatter partitioned chunks.

        Args:
            rank_tensors: Mapping from rank ID to the full tensor from that rank.
            op: Reduction operation ('sum' or 'mean').
            dim: Dimension along which to split and scatter the reduced tensor.

        Returns:
            Mapping from rank ID to that rank's scattered chunk.
        """
        expected_ranks = set(self.grid.ranks)
        actual_ranks = set(rank_tensors.keys())

        if actual_ranks != expected_ranks:
            missing = expected_ranks - actual_ranks
            extra = actual_ranks - expected_ranks
            raise ValueError(
                f"Tensors must be provided for all ranks. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )

        first_tensor = rank_tensors[self.grid.ranks[0]]
        for rank in self.grid.ranks[1:]:
            t = rank_tensors[rank]
            if t.shape != first_tensor.shape:
                raise ValueError(
                    f"Shape mismatch at rank {rank}: expected {first_tensor.shape}, got {t.shape}"
                )
            if t.dtype != first_tensor.dtype:
                raise ValueError(
                    f"Dtype mismatch at rank {rank}: expected {first_tensor.dtype}, got {t.dtype}"
                )

        dim_size = first_tensor.size(dim)
        if dim_size % self.grid.world_size != 0:
            raise ValueError(
                f"Dimension {dim} size {dim_size} is not evenly divisible by world_size {self.grid.world_size}"
            )

        # 1. Reduce
        accumulated = torch.zeros_like(first_tensor)
        for rank in self.grid.ranks:
            accumulated = accumulated + rank_tensors[rank]

        if op == "mean":
            reduced = accumulated / float(self.grid.world_size)
        elif op == "sum":
            reduced = accumulated
        else:
            raise ValueError(f"Unsupported reduction op: '{op}', expected 'sum' or 'mean'")

        # 2. Scatter along dim
        chunks = list(torch.chunk(reduced, chunks=self.grid.world_size, dim=dim))

        self.barrier()
        return {rank: chunks[rank].clone() for rank in self.grid.ranks}
