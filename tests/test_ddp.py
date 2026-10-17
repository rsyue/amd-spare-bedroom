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


def test_virtual_ddp_gradient_sanitization_non_finite() -> None:
    """Verify that non-finite gradients (inf / nan) are safely sanitized during All-Reduce."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")
    model = SimpleModel()
    ddp: VirtualDDP[SimpleModel] = VirtualDDP(model, grid=grid)

    # Simulate forward/backward on rank 0 with valid grad
    out0 = ddp.forward(rank=0, x=torch.ones((1, 4)))
    loss0 = out0.sum()
    loss0.backward()

    # Simulate forward/backward on rank 1 with valid grad
    out1 = ddp.forward(rank=1, x=torch.ones((1, 4)))
    loss1 = out1.sum()
    loss1.backward()

    # Inject inf and nan into rank 1 gradients
    p1 = ddp.get_replica(1).fc.weight
    assert p1.grad is not None
    p1.grad[0, 0] = float("-inf")
    p1.grad[0, 1] = float("nan")

    # Gradient sync must succeed without crashing or propagating NaNs/Infs
    ddp.sync_gradients()

    grad0 = ddp.get_replica(0).fc.weight.grad
    grad1 = ddp.get_replica(1).fc.weight.grad
    assert grad0 is not None and grad1 is not None
    assert not torch.isnan(grad0).any()
    assert not torch.isinf(grad0).any()
    assert torch.allclose(grad0, grad1)


class DropoutModel(nn.Module):
    """Model with stochastic dropout for verifying train/eval mode switching."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.dropout(self.fc(x)))


def test_virtual_ddp_train_eval_mode_and_eval_forward() -> None:
    """Verify .train(), .eval(), and .eval_forward() disable dropout and gradient tracking."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu", seed=42)
    model = DropoutModel()
    ddp: VirtualDDP[DropoutModel] = VirtualDDP(model, grid=grid)

    assert ddp.is_training is True
    x = torch.ones((4, 8))

    # Switch to eval mode
    returned = ddp.eval()
    assert returned is ddp
    assert ddp.is_training is False
    for rank in grid.ranks:
        assert ddp.get_replica(rank).training is False

    # eval_forward must disable grad tracking and produce deterministic outputs without dropout
    out_a = ddp.eval_forward(rank=0, x=x)
    out_b = ddp.eval_forward(rank=0, x=x)
    assert out_a.requires_grad is False
    assert torch.equal(out_a, out_b)

    # Switch back to train mode: repeated forward passes with dropout yield distinct outputs
    ddp.train()
    assert ddp.is_training is True
    for rank in grid.ranks:
        assert ddp.get_replica(rank).training is True
    out_train_a = ddp.forward(rank=0, x=x)
    out_train_b = ddp.forward(rank=0, x=x)
    assert out_train_a.requires_grad is True
    assert not torch.equal(out_train_a, out_train_b)


def test_virtual_ddp_clip_grad_norm_and_step() -> None:
    """Verify VirtualDDP.clip_grad_norm bounds gradient L2 norm and step accepts max_grad_norm."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu", seed=42)
    model = SimpleModel()
    ddp: VirtualDDP[SimpleModel] = VirtualDDP(model, grid=grid)

    x0 = torch.full((2, 4), 10.0)
    y0 = torch.zeros((2, 2))
    x1 = torch.full((2, 4), 12.0)
    y1 = torch.zeros((2, 2))

    nn.functional.mse_loss(ddp.forward(0, x0), y0).backward()
    nn.functional.mse_loss(ddp.forward(1, x1), y1).backward()
    ddp.sync_gradients()

    norms = ddp.clip_grad_norm(max_norm=0.5)
    assert set(norms.keys()) == {0, 1}
    assert norms[0] > 0.5
    assert abs(norms[0] - norms[1]) < 1e-6

    for rank in grid.ranks:
        clipped_sq = sum(
            float(p.grad.norm(2).item() ** 2)
            for p in ddp.get_replica(rank).parameters()
            if p.grad is not None
        )
        assert (clipped_sq**0.5) <= 0.5 + 1e-4

    optimizers = ddp.create_optimizers(torch.optim.SGD, lr=0.05)
    ddp.step(optimizers)
    assert ddp.assert_parameters_synced()

    # Also verify step(..., max_grad_norm=0.5) directly clips and returns norms
    nn.functional.mse_loss(ddp.forward(0, x0), y0).backward()
    nn.functional.mse_loss(ddp.forward(1, x1), y1).backward()
    ddp.sync_gradients()
    step_norms = ddp.step(optimizers, max_grad_norm=0.5)
    assert step_norms is not None
    assert step_norms[0] > 0.5
    assert ddp.assert_parameters_synced()


def test_virtual_ddp_amp_bfloat16_and_float16() -> None:
    """Verify VirtualDDP supports bfloat16/float16 autocast, unscaling, and overflow backoff."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu", seed=42)
    base_model = SimpleModel()
    x0 = torch.randn(2, 4)
    y0 = torch.randn(2, 2)
    x1 = torch.randn(2, 4)
    y1 = torch.randn(2, 2)

    # Reference FP32 gradient norm
    ddp_fp32: VirtualDDP[SimpleModel] = VirtualDDP(copy.deepcopy(base_model), grid=grid)
    ddp_fp32.backward(0, nn.functional.mse_loss(ddp_fp32.forward(0, x0), y0))
    ddp_fp32.backward(1, nn.functional.mse_loss(ddp_fp32.forward(1, x1), y1))
    ddp_fp32.sync_gradients()
    fp32_norm = ddp_fp32.clip_grad_norm(max_norm=10.0)[0]

    for precision, expected_dtype in [("bfloat16", torch.bfloat16), ("float16", torch.float16)]:
        ddp_amp: VirtualDDP[SimpleModel] = VirtualDDP(
            copy.deepcopy(base_model), grid=grid, amp_dtype=precision
        )
        assert ddp_amp.amp_dtype == expected_dtype
        optimizers = ddp_amp.create_optimizers(torch.optim.SGD, lr=0.05)

        out0 = ddp_amp.forward(0, x0)
        out1 = ddp_amp.forward(1, x1)
        assert out0.dtype == expected_dtype
        assert out1.dtype == expected_dtype

        eval_out = ddp_amp.eval_forward(0, x0)
        assert eval_out.dtype == expected_dtype
        assert eval_out.requires_grad is False

        loss0 = nn.functional.mse_loss(out0, y0)
        loss1 = nn.functional.mse_loss(out1, y1)
        ddp_amp.backward(0, loss0)
        ddp_amp.backward(1, loss1)
        ddp_amp.sync_gradients()

        amp_norms = ddp_amp.clip_grad_norm(max_norm=10.0)
        # GradScaler for float16 must be unscaled prior to computing/clipping norm
        assert abs(amp_norms[0] - fp32_norm) < 0.05

        ddp_amp.step(optimizers)
        assert ddp_amp.assert_parameters_synced()

    # Verify FP16 overflow skips parameter update and backs off GradScaler scale factor
    ddp_fp16: VirtualDDP[SimpleModel] = VirtualDDP(
        copy.deepcopy(base_model), grid=grid, amp_dtype="float16"
    )
    opts_fp16 = ddp_fp16.create_optimizers(torch.optim.SGD, lr=0.1)
    initial_scale = ddp_fp16.get_grad_scale(0)
    weights_before = ddp_fp16.get_replica(0).fc.weight.data.clone()

    ddp_fp16.backward(0, nn.functional.mse_loss(ddp_fp16.forward(0, x0), y0))
    ddp_fp16.backward(1, nn.functional.mse_loss(ddp_fp16.forward(1, x1), y1))
    grad0 = ddp_fp16.get_replica(0).fc.weight.grad
    assert grad0 is not None
    grad0[0, 0] = float("inf")

    ddp_fp16.sync_gradients()
    ddp_fp16.step(opts_fp16)

    assert ddp_fp16.get_grad_scale(0) == initial_scale * 0.5
    assert torch.allclose(ddp_fp16.get_replica(0).fc.weight.data, weights_before)
