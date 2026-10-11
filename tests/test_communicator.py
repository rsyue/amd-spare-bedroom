"""Tests for Seam 2: VirtualCommunicator and LockstepBarrier collective operations."""

import pytest
import torch

from amd_spare_bedroom.communicator import VirtualCommunicator
from amd_spare_bedroom.grid import VirtualGrid


def test_communicator_all_reduce_mean() -> None:
    """Verify that all_reduce with op='mean' synchronizes tensors across all ranks."""
    grid = VirtualGrid.create(world_size=4, total_memory_gb=16.0, device_type="cpu")
    comm = VirtualCommunicator(grid)

    rank_tensors = {
        0: torch.tensor([1.0, 2.0]),
        1: torch.tensor([3.0, 4.0]),
        2: torch.tensor([5.0, 6.0]),
        3: torch.tensor([7.0, 8.0]),
    }

    reduced = comm.all_reduce(rank_tensors, op="mean")

    expected = torch.tensor([4.0, 5.0])
    for rank in range(4):
        assert rank in reduced
        assert torch.allclose(reduced[rank], expected)
        # Ensure returned tensors are distinct instances, not aliased references
        assert reduced[rank] is not rank_tensors[rank]


def test_communicator_all_reduce_sum() -> None:
    """Verify that all_reduce with op='sum' accumulates values across all ranks."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")
    comm = VirtualCommunicator(grid)

    rank_tensors = {
        0: torch.tensor([10.0, 20.0]),
        1: torch.tensor([30.0, 40.0]),
    }

    reduced = comm.all_reduce(rank_tensors, op="sum")

    expected = torch.tensor([40.0, 60.0])
    for rank in range(2):
        assert torch.allclose(reduced[rank], expected)


def test_communicator_all_reduce_mismatched_ranks() -> None:
    """Verify that missing ranks in all_reduce raises ValueError."""
    grid = VirtualGrid.create(world_size=3, total_memory_gb=12.0, device_type="cpu")
    comm = VirtualCommunicator(grid)

    # Missing rank 2
    rank_tensors = {
        0: torch.tensor([1.0]),
        1: torch.tensor([2.0]),
    }

    with pytest.raises(ValueError, match="Tensors must be provided for all ranks"):
        comm.all_reduce(rank_tensors, op="sum")


def test_communicator_all_reduce_mismatched_shapes() -> None:
    """Verify that incompatible tensor shapes across ranks raise ValueError."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")
    comm = VirtualCommunicator(grid)

    rank_tensors = {
        0: torch.tensor([1.0, 2.0]),
        1: torch.tensor([3.0, 4.0, 5.0]),
    }

    with pytest.raises(ValueError, match="Shape mismatch"):
        comm.all_reduce(rank_tensors, op="mean")


def test_communicator_broadcast() -> None:
    """Verify that broadcast copies the source rank tensor to all ranks."""
    grid = VirtualGrid.create(world_size=3, total_memory_gb=12.0, device_type="cpu")
    comm = VirtualCommunicator(grid)

    src_tensor = torch.tensor([42.0, 99.0])
    broadcasted = comm.broadcast(src_rank=1, tensor=src_tensor)

    for rank in range(3):
        assert torch.allclose(broadcasted[rank], src_tensor)
        if rank != 1:
            assert broadcasted[rank] is not src_tensor


def test_communicator_barrier() -> None:
    """Verify lockstep barrier execution."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")
    comm = VirtualCommunicator(grid)

    comm.barrier()
    assert comm.barrier_count == 1
