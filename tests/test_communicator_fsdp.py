"""Tests for Seam 5A: All-Gather and Reduce-Scatter collective communication primitives."""

import pytest
import torch

from amd_spare_bedroom.communicator import VirtualCommunicator
from amd_spare_bedroom.grid import VirtualGrid


def test_all_gather_1d_tensors() -> None:
    """Verify that all_gather concatenates rank shards and distributes full tensor to all ranks."""
    grid = VirtualGrid.create(world_size=4, total_memory_gb=16.0, device_type="cpu")
    comm = VirtualCommunicator(grid)

    shards = {
        0: torch.tensor([0.0, 1.0]),
        1: torch.tensor([10.0, 11.0]),
        2: torch.tensor([20.0, 21.0]),
        3: torch.tensor([30.0, 31.0]),
    }

    gathered = comm.all_gather(shards, dim=0)

    expected = torch.tensor([0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0])
    for rank in range(4):
        assert rank in gathered
        assert torch.allclose(gathered[rank], expected)
        assert gathered[rank] is not shards[rank]


def test_all_gather_2d_tensors() -> None:
    """Verify all_gather across 2D tensors along dim=0."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")
    comm = VirtualCommunicator(grid)

    shards = {
        0: torch.ones((2, 3)),
        1: torch.full((2, 3), 2.0),
    }

    gathered = comm.all_gather(shards, dim=0)

    for rank in range(2):
        assert gathered[rank].shape == (4, 3)
        assert torch.allclose(gathered[rank][:2], torch.ones((2, 3)))
        assert torch.allclose(gathered[rank][2:], torch.full((2, 3), 2.0))


def test_reduce_scatter_mean() -> None:
    """Verify reduce_scatter computes mean reduction and scatters partitioned chunks."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")
    comm = VirtualCommunicator(grid)

    tensors = {
        0: torch.tensor([2.0, 4.0, 6.0, 8.0]),
        1: torch.tensor([4.0, 8.0, 12.0, 16.0]),
    }

    scattered = comm.reduce_scatter(tensors, op="mean", dim=0)

    # Mean is [3.0, 6.0, 9.0, 12.0] -> Rank 0 gets [3.0, 6.0], Rank 1 gets [9.0, 12.0]
    expected_0 = torch.tensor([3.0, 6.0])
    expected_1 = torch.tensor([9.0, 12.0])

    assert torch.allclose(scattered[0], expected_0)
    assert torch.allclose(scattered[1], expected_1)


def test_reduce_scatter_sum() -> None:
    """Verify reduce_scatter computes sum reduction and scatters partitioned chunks."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")
    comm = VirtualCommunicator(grid)

    tensors = {
        0: torch.tensor([1.0, 2.0]),
        1: torch.tensor([3.0, 4.0]),
    }

    scattered = comm.reduce_scatter(tensors, op="sum", dim=0)

    # Sum is [4.0, 6.0] -> Rank 0 gets [4.0], Rank 1 gets [6.0]
    assert torch.allclose(scattered[0], torch.tensor([4.0]))
    assert torch.allclose(scattered[1], torch.tensor([6.0]))


def test_all_gather_missing_rank_raises() -> None:
    """Verify missing ranks in all_gather raises ValueError."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")
    comm = VirtualCommunicator(grid)

    shards = {0: torch.tensor([1.0])}
    with pytest.raises(ValueError, match="Tensors must be provided for all ranks"):
        comm.all_gather(shards)


def test_reduce_scatter_non_divisible_raises() -> None:
    """Verify reduce_scatter raises ValueError when dimension is not divisible by world size."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")
    comm = VirtualCommunicator(grid)

    tensors = {
        0: torch.tensor([1.0, 2.0, 3.0]),
        1: torch.tensor([4.0, 5.0, 6.0]),
    }
    with pytest.raises(ValueError, match="not evenly divisible"):
        comm.reduce_scatter(tensors, dim=0)
