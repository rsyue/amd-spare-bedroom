"""Tests for Seam 5B: VirtualFSDP parameter sharding, all-gather, and reduce-scatter."""

import copy
from typing import cast

import torch
import torch.nn as nn

from amd_spare_bedroom.ddp import VirtualDDP
from amd_spare_bedroom.fsdp import VirtualFSDP
from amd_spare_bedroom.grid import VirtualGrid


class SimpleLinear(nn.Module):
    """Linear model for testing FSDP sharding."""

    def __init__(self) -> None:
        super().__init__()
        self.fc: nn.Linear = nn.Linear(8, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc(x))


def test_fsdp_parameter_sharding_memory() -> None:
    """Verify that parameters are evenly sharded across virtual ranks, cutting per-rank storage."""
    grid = VirtualGrid.create(world_size=4, total_memory_gb=16.0, device_type="cpu")
    model = SimpleLinear()

    # Total parameters: 8*4 + 4 = 36 elements
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 36

    fsdp = VirtualFSDP(model, grid=grid)

    # Each rank must store 36 / 4 = 9 elements
    for rank in range(4):
        rank_elements = fsdp.get_rank_parameter_numel(rank)
        assert rank_elements == 9

    # Reconstructed full parameters must match original weights
    full_params = fsdp.get_full_parameters()
    orig_params = {name: p.data for name, p in model.named_parameters()}
    for name in orig_params:
        assert torch.allclose(full_params[name], orig_params[name])


def test_fsdp_forward_and_all_gather() -> None:
    """Verify that all-gather accurately reconstructs full weights for rank forward passes."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")
    model = SimpleLinear()
    fsdp = VirtualFSDP(model, grid=grid)

    x = torch.randn(2, 8)
    expected_out = model(x)

    # Forward pass on rank 0
    out0 = fsdp.forward(rank=0, x=x)
    assert torch.allclose(out0, expected_out, atol=1e-6)


def test_fsdp_reduce_scatter_gradient_and_step() -> None:
    """Verify reduce-scatter accumulates gradients into sharded rank parameters and steps."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")
    model = SimpleLinear()
    fsdp = VirtualFSDP(model, grid=grid)

    x0 = torch.randn(2, 8)
    y0 = torch.randn(2, 4)
    x1 = torch.randn(2, 8)
    y1 = torch.randn(2, 4)

    # Rank 0 forward & backward
    out0 = fsdp.forward(rank=0, x=x0)
    loss0 = nn.functional.mse_loss(out0, y0)
    loss0.backward()

    # Rank 1 forward & backward
    out1 = fsdp.forward(rank=1, x=x1)
    loss1 = nn.functional.mse_loss(out1, y1)
    loss1.backward()

    # Reduce-scatter gradients back to rank shards
    fsdp.sync_gradients()

    # Verify that each rank holds gradients for its sharded parameter slice
    for rank in range(2):
        for shard in fsdp.get_sharded_parameters(rank):
            assert shard.grad is not None

    # Step optimizers on sharded parameters
    optimizers = fsdp.create_optimizers(torch.optim.SGD, lr=0.1)
    fsdp.step(optimizers)

    # After step, gradients should be reset
    for rank in range(2):
        for shard in fsdp.get_sharded_parameters(rank):
            assert shard.grad is None or torch.count_nonzero(shard.grad) == 0


def test_fsdp_mathematical_equivalence_to_ddp() -> None:
    """Verify that VirtualFSDP produces the exact same parameter updates as VirtualDDP."""
    torch.manual_seed(42)
    base_model = SimpleLinear()

    # Deep copies for FSDP and DDP
    fsdp_model = copy.deepcopy(base_model)
    ddp_model = copy.deepcopy(base_model)

    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")

    fsdp = VirtualFSDP(fsdp_model, grid=grid)
    ddp: VirtualDDP[SimpleLinear] = VirtualDDP(ddp_model, grid=grid)

    x0 = torch.randn(2, 8)
    y0 = torch.randn(2, 4)
    x1 = torch.randn(2, 8)
    y1 = torch.randn(2, 4)

    # 1. FSDP forward, backward, sync
    f_out0 = fsdp.forward(rank=0, x=x0)
    f_loss0 = nn.functional.mse_loss(f_out0, y0)
    f_loss0.backward()

    f_out1 = fsdp.forward(rank=1, x=x1)
    f_loss1 = nn.functional.mse_loss(f_out1, y1)
    f_loss1.backward()

    fsdp.sync_gradients()
    fsdp_opts = fsdp.create_optimizers(torch.optim.SGD, lr=0.05)
    fsdp.step(fsdp_opts)

    # 2. DDP forward, backward, sync
    d_out0 = ddp.forward(rank=0, x=x0)
    d_loss0 = nn.functional.mse_loss(d_out0, y0)
    d_loss0.backward()

    d_out1 = ddp.forward(rank=1, x=x1)
    d_loss1 = nn.functional.mse_loss(d_out1, y1)
    d_loss1.backward()

    ddp.sync_gradients()
    ddp_opts = ddp.create_optimizers(torch.optim.SGD, lr=0.05)
    ddp.step(ddp_opts)

    # 3. Compare updated weights between FSDP and DDP
    fsdp_full_weights = fsdp.get_full_parameters()
    ddp_replica_0 = ddp.get_replica(0)

    for name, p_ddp in ddp_replica_0.named_parameters():
        p_fsdp = fsdp_full_weights[name]
        assert torch.allclose(p_fsdp, p_ddp.data, atol=1e-6)
