"""Tests for Seam 1: VirtualGrid and VirtualDevice topology."""

import random

import numpy as np
import pytest
import torch

from amd_spare_bedroom import resolve_amp_dtype, seed_everything
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


def test_virtual_grid_auto_device_detection() -> None:
    """Verify that device_type='auto' resolves to cpu or cuda."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=16.0, device_type="auto")
    assert grid.device_type in ("cpu", "cuda")


def test_seed_everything_deterministic_across_rngs() -> None:
    """Verify seed_everything produces reproducible Python, NumPy, and PyTorch draws."""
    returned = seed_everything(1337)
    assert returned == 1337
    py_a = [random.random() for _ in range(4)]
    np_a = np.random.rand(4)
    th_a = torch.randn(4)

    seed_everything(1337)
    py_b = [random.random() for _ in range(4)]
    np_b = np.random.rand(4)
    th_b = torch.randn(4)

    assert py_a == py_b
    assert np.allclose(np_a, np_b)
    assert torch.allclose(th_a, th_b)


def test_virtual_grid_seed_parameter_and_per_rank_generators() -> None:
    """Verify VirtualGrid.create(..., seed=...) seeds global and per-rank RNG generators."""
    from amd_spare_bedroom import VirtualDDP, VirtualFSDP

    grid1 = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu", seed=2026)
    m1_ddp = VirtualDDP(torch.nn.Linear(4, 2), grid=grid1)
    m1_fsdp = VirtualFSDP(torch.nn.Linear(4, 2), grid=grid1)
    r0_a = torch.randn(4, generator=grid1.get_generator(0))
    r1_a = torch.randn(4, generator=grid1.get_generator(1))

    grid2 = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu", seed=2026)
    m2_ddp = VirtualDDP(torch.nn.Linear(4, 2), grid=grid2)
    m2_fsdp = VirtualFSDP(torch.nn.Linear(4, 2), grid=grid2)
    r0_b = torch.randn(4, generator=grid2.get_generator(0))
    r1_b = torch.randn(4, generator=grid2.get_generator(1))

    assert grid1.seed == 2026
    assert torch.allclose(r0_a, r0_b)
    assert torch.allclose(r1_a, r1_b)
    assert not torch.allclose(r0_a, r1_a)

    for p1, p2 in zip(
        m1_ddp.get_replica(0).parameters(), m2_ddp.get_replica(0).parameters(), strict=True
    ):
        assert torch.allclose(p1, p2)

    full1 = m1_fsdp.get_full_parameters()
    full2 = m2_fsdp.get_full_parameters()
    for name in full1:
        assert torch.allclose(full1[name], full2[name])

    with pytest.raises(IndexError, match="Rank 2 out of bounds"):
        grid1.get_generator(2)


def test_resolve_amp_dtype_variants() -> None:
    """Verify resolve_amp_dtype normalizes string aliases and torch.dtype values."""
    assert resolve_amp_dtype(None) is None
    assert resolve_amp_dtype("none") is None
    assert resolve_amp_dtype("fp32") is None
    assert resolve_amp_dtype("float32") is None
    assert resolve_amp_dtype(torch.float32) is None
    assert resolve_amp_dtype("bf16") == torch.bfloat16
    assert resolve_amp_dtype("bfloat16") == torch.bfloat16
    assert resolve_amp_dtype(torch.bfloat16) == torch.bfloat16
    assert resolve_amp_dtype("fp16") == torch.float16
    assert resolve_amp_dtype("float16") == torch.float16
    assert resolve_amp_dtype(torch.float16) == torch.float16

    with pytest.raises(ValueError, match="Unsupported AMP dtype"):
        resolve_amp_dtype("int8")


def test_seed_everything_bounds_validation() -> None:
    """Verify seed_everything rejects negative seeds, floats, booleans, and 32-bit overflow."""
    with pytest.raises(ValueError, match="Seed must be an integer between 0 and 4294967295"):
        seed_everything(-1)

    with pytest.raises(ValueError, match="Seed must be an integer between 0 and 4294967295"):
        seed_everything(2**32)

    with pytest.raises(ValueError, match="Seed must be an integer between 0 and 4294967295"):
        seed_everything(42.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Seed must be an integer between 0 and 4294967295"):
        seed_everything(True)


def test_virtual_grid_memory_validations() -> None:
    """Verify VirtualGrid rejects zero or negative total memory allocations."""
    with pytest.raises(ValueError, match="total_memory_bytes must be > 0"):
        VirtualGrid(world_size=2, total_memory_bytes=0, device_type="cpu")

    with pytest.raises(ValueError, match="total_memory_bytes must be > 0"):
        VirtualGrid(world_size=2, total_memory_bytes=-1024, device_type="cpu")


def test_virtual_grid_generator_device_alignment() -> None:
    """Verify that per-rank Generators match the grid's target device type."""
    grid_cpu = VirtualGrid.create(world_size=2, total_memory_gb=4.0, device_type="cpu", seed=42)
    assert grid_cpu.get_generator(0).device == torch.device("cpu")
    assert grid_cpu.get_generator(1).device == torch.device("cpu")

    if torch.cuda.is_available():
        grid_cuda = VirtualGrid.create(
            world_size=2, total_memory_gb=4.0, device_type="cuda", seed=42
        )
        assert grid_cuda.get_generator(0).device == torch.device("cuda")
