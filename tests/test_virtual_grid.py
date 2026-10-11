"""Tests for Seam 1: VirtualGrid and VirtualDevice topology."""

import pytest

from amd_spare_bedroom.grid import VirtualDevice, VirtualGrid


def test_virtual_grid_creation_and_partitioning() -> None:
    """Verify that a VirtualGrid correctly divides total memory among emulated ranks."""
    world_size = 4
    total_memory_gb = 96.0

    grid = VirtualGrid.create(
        world_size=world_size,
        total_memory_gb=total_memory_gb,
        device_type="cpu",
    )

    assert grid.world_size == 4
    assert grid.ranks == [0, 1, 2, 3]

    expected_total_bytes = int(96.0 * 1024 * 1024 * 1024)
    expected_per_rank_bytes = expected_total_bytes // 4

    assert grid.total_memory_bytes == expected_total_bytes
    assert grid.per_rank_memory_bytes == expected_per_rank_bytes

    # Check each rank device
    for rank in range(world_size):
        device = grid.get_device(rank)
        assert isinstance(device, VirtualDevice)
        assert device.rank == rank
        assert device.memory_limit_bytes == expected_per_rank_bytes
        assert device.device_type == "cpu"
        assert device.address_space_id == f"vdev:{rank}"


def test_virtual_grid_rank_out_of_bounds() -> None:
    """Verify that querying an invalid rank raises IndexError."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=16.0, device_type="cpu")

    with pytest.raises(IndexError, match="Rank 2 out of bounds"):
        grid.get_device(2)

    with pytest.raises(IndexError, match="Rank -1 out of bounds"):
        grid.get_device(-1)


def test_virtual_grid_invalid_world_size() -> None:
    """Verify that world_size < 1 is rejected."""
    with pytest.raises(ValueError, match="world_size must be >= 1"):
        VirtualGrid.create(world_size=0, total_memory_gb=16.0, device_type="cpu")
