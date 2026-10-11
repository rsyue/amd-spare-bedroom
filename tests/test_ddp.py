"""Tests for Seam 4: VirtualDDP model replication and gradient synchronization."""

import copy
from typing import cast

import torch
import torch.nn as nn

from amd_spare_bedroom.ddp import VirtualDDP
from amd_spare_bedroom.grid import VirtualGrid


class SimpleModel(nn.Module):
    """Simple linear model for testing distributed mechanics."""

    def __init__(self) -> None:
        super().__init__()
        self.fc: nn.Linear = nn.Linear(4, 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc(x))


def test_virtual_ddp_initial_parameter_sync() -> None:
    """Verify that all rank replicas are initialized with identical weights."""
    grid = VirtualGrid.create(world_size=3, total_memory_gb=12.0, device_type="cpu")
    model = SimpleModel()

    ddp: VirtualDDP[SimpleModel] = VirtualDDP(model, grid=grid)

    assert len(ddp.replicas) == 3
    # Check that parameters are identical across replicas
    for rank in range(3):
        replica = ddp.get_replica(rank)
        for p_orig, p_rep in zip(model.parameters(), replica.parameters(), strict=True):
            assert torch.allclose(p_orig, p_rep)
            assert p_orig is not p_rep  # Ensure memory isolation (distinct copy)


def test_virtual_ddp_gradient_sync_and_step() -> None:
    """Verify gradient all-reduce and lockstep parameter synchronization across ranks."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")
    model = SimpleModel()
    ddp: VirtualDDP[SimpleModel] = VirtualDDP(model, grid=grid)

    x0 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    y0 = torch.tensor([[1.0, 0.0]])

    x1 = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    y1 = torch.tensor([[0.0, 1.0]])

    # Forward pass on rank 0
    out0 = ddp.forward(rank=0, x=x0)
    loss0 = nn.functional.mse_loss(out0, y0)
    loss0.backward()

    # Forward pass on rank 1
    out1 = ddp.forward(rank=1, x=x1)
    loss1 = nn.functional.mse_loss(out1, y1)
    loss1.backward()

    # Verify that prior to sync, local gradients differ
    grad0 = ddp.get_replica(0).fc.weight.grad
    grad1 = ddp.get_replica(1).fc.weight.grad
    assert grad0 is not None and grad1 is not None
    assert not torch.allclose(grad0, grad1)

    expected_avg_grad = (grad0 + grad1) / 2.0

    # Execute gradient all-reduce
    ddp.sync_gradients()

    # Gradients must now be synchronized to the exact average
    final_grad0 = ddp.get_replica(0).fc.weight.grad
    final_grad1 = ddp.get_replica(1).fc.weight.grad
    assert final_grad0 is not None and final_grad1 is not None
    assert torch.allclose(final_grad0, expected_avg_grad)
    assert torch.allclose(final_grad1, expected_avg_grad)

    # Step optimizers
    optimizers = ddp.create_optimizers(torch.optim.SGD, lr=0.1)
    ddp.step(optimizers)

    # Replicas must remain in exact parameter lockstep
    assert ddp.assert_parameters_synced()


def test_virtual_ddp_mathematical_equivalence_to_full_batch() -> None:
    """Verify that VirtualDDP across N ranks computes the exact same gradient as full batch."""
    torch.manual_seed(42)
    base_model = SimpleModel()

    # Create two identical copies: one for VirtualDDP (world_size=2) and one for full batch
    full_batch_model = copy.deepcopy(base_model)
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")
    ddp: VirtualDDP[SimpleModel] = VirtualDDP(base_model, grid=grid)

    x0 = torch.randn(1, 4)
    y0 = torch.randn(1, 2)
    x1 = torch.randn(1, 4)
    y1 = torch.randn(1, 2)

    # 1. VirtualDDP computation (batch size 1 per rank)
    out0 = ddp.forward(rank=0, x=x0)
    loss0 = nn.functional.mse_loss(out0, y0)
    loss0.backward()

    out1 = ddp.forward(rank=1, x=x1)
    loss1 = nn.functional.mse_loss(out1, y1)
    loss1.backward()

    ddp.sync_gradients()

    # 2. Full batch computation (batch size 2)
    x_full = torch.cat([x0, x1], dim=0)
    y_full = torch.cat([y0, y1], dim=0)
    out_full = full_batch_model(x_full)
    loss_full = nn.functional.mse_loss(out_full, y_full)
    loss_full.backward()

    # Both must match exactly
    ddp_grad = ddp.get_replica(0).fc.weight.grad
    full_grad = full_batch_model.fc.weight.grad
    assert ddp_grad is not None and full_grad is not None
    assert torch.allclose(ddp_grad, full_grad, atol=1e-6)
