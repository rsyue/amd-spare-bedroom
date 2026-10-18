"""Tests for Seam 5B: VirtualFSDP parameter sharding, all-gather, and reduce-scatter."""

import copy
import math
from typing import cast

import pytest
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


def test_virtual_fsdp_gradient_sanitization_non_finite() -> None:
    """Verify that non-finite gradients (inf / nan) are safely sanitized during Reduce-Scatter."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu")
    model = SimpleLinear()
    fsdp = VirtualFSDP(model, grid=grid)

    # Forward on rank 0
    out0 = fsdp.forward(rank=0, x=torch.ones((2, 8)))
    loss0 = out0.sum()
    loss0.backward()

    # Forward on rank 1
    out1 = fsdp.forward(rank=1, x=torch.ones((2, 8)))
    loss1 = out1.sum()
    loss1.backward()

    # Inject inf and nan into rank 1 gradients
    p1 = dict(fsdp._exec_replicas[1].named_parameters())["fc.weight"]
    assert p1.grad is not None
    p1.grad[0, 0] = float("-inf")
    p1.grad[0, 1] = float("nan")

    # Gradient sync must succeed and shards must have finite values
    fsdp.sync_gradients()

    for rank in grid.ranks:
        for shard_param in fsdp.get_sharded_parameters(rank):
            assert shard_param.grad is not None
            assert not torch.isnan(shard_param.grad).any()
            assert not torch.isinf(shard_param.grad).any()


class DropoutLinear(nn.Module):
    """Model with stochastic dropout for verifying FSDP train/eval mode switching."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.dropout(self.fc(x)))


def test_virtual_fsdp_train_eval_mode_and_eval_forward() -> None:
    """Verify .train(), .eval(), and .eval_forward() disable dropout and gradient tracking in FSDP."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu", seed=42)
    model = DropoutLinear()
    fsdp: VirtualFSDP[DropoutLinear] = VirtualFSDP(model, grid=grid)

    assert fsdp.is_training is True
    x = torch.ones((4, 8))

    returned = fsdp.eval()
    assert returned is fsdp
    assert fsdp.is_training is False

    out_a = fsdp.eval_forward(rank=0, x=x)
    out_b = fsdp.eval_forward(rank=0, x=x)
    assert out_a.requires_grad is False
    assert torch.equal(out_a, out_b)

    fsdp.train()
    assert fsdp.is_training is True
    out_train_a = fsdp.forward(rank=0, x=x)
    out_train_b = fsdp.forward(rank=0, x=x)
    assert out_train_a.requires_grad is True
    assert not torch.equal(out_train_a, out_train_b)


def test_virtual_fsdp_clip_grad_norm_matches_ddp_global_norm() -> None:
    """Verify FSDP clip_grad_norm computes exact global L2/inf/zero norm across shards matching DDP."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu", seed=123)
    base_model = SimpleLinear()

    fsdp = VirtualFSDP(copy.deepcopy(base_model), grid=grid)
    ddp: VirtualDDP[SimpleLinear] = VirtualDDP(copy.deepcopy(base_model), grid=grid)

    x0 = torch.randn(3, 8) * 5.0
    y0 = torch.randn(3, 4) * 5.0
    x1 = torch.randn(3, 8) * 5.0
    y1 = torch.randn(3, 4) * 5.0

    # FSDP forward/backward/sync
    nn.functional.mse_loss(fsdp.forward(0, x0), y0).backward()
    nn.functional.mse_loss(fsdp.forward(1, x1), y1).backward()
    fsdp.sync_gradients()

    # DDP forward/backward/sync
    nn.functional.mse_loss(ddp.forward(0, x0), y0).backward()
    nn.functional.mse_loss(ddp.forward(1, x1), y1).backward()
    ddp.sync_gradients()

    # Clip both at max_norm=0.25
    fsdp_norms = fsdp.clip_grad_norm(max_norm=0.25)
    ddp_norms = ddp.clip_grad_norm(max_norm=0.25)

    assert fsdp_norms[0] > 0.25
    assert abs(fsdp_norms[0] - ddp_norms[0]) < 1e-5
    assert abs(fsdp_norms[1] - ddp_norms[1]) < 1e-5

    # Step both and verify weights remain identical after global norm clipping
    fsdp_opts = fsdp.create_optimizers(torch.optim.SGD, lr=0.1)
    ddp_opts = ddp.create_optimizers(torch.optim.SGD, lr=0.1)
    fsdp.step(fsdp_opts)
    ddp.step(ddp_opts)

    fsdp_full = fsdp.get_full_parameters()
    for name, p_ddp in ddp.get_replica(0).named_parameters():
        assert torch.allclose(fsdp_full[name], p_ddp.data, atol=1e-6)

    # Verify inf-norm and max_norm=0.0 parity
    nn.functional.mse_loss(fsdp.forward(0, x0), y0).backward()
    nn.functional.mse_loss(fsdp.forward(1, x1), y1).backward()
    fsdp.sync_gradients()
    nn.functional.mse_loss(ddp.forward(0, x0), y0).backward()
    nn.functional.mse_loss(ddp.forward(1, x1), y1).backward()
    ddp.sync_gradients()

    fsdp_inf = fsdp.clip_grad_norm(max_norm=0.0, norm_type=float("inf"))
    ddp_inf = ddp.clip_grad_norm(max_norm=0.0, norm_type=float("inf"))
    assert abs(fsdp_inf[0] - ddp_inf[0]) < 1e-5
    for rank in grid.ranks:
        for shard in fsdp.get_sharded_parameters(rank):
            assert shard.grad is not None
            assert torch.count_nonzero(shard.grad) == 0


def test_virtual_fsdp_amp_bfloat16_and_float16() -> None:
    """Verify VirtualFSDP supports bfloat16/float16 autocast, unscaling, and overflow backoff."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=8.0, device_type="cpu", seed=42)
    base_model = SimpleLinear()
    x0 = torch.randn(2, 8)
    y0 = torch.randn(2, 4)
    x1 = torch.randn(2, 8)
    y1 = torch.randn(2, 4)

    # Reference FP32 gradient norm
    fsdp_fp32 = VirtualFSDP(copy.deepcopy(base_model), grid=grid)
    fsdp_fp32.backward(0, nn.functional.mse_loss(fsdp_fp32.forward(0, x0), y0))
    fsdp_fp32.backward(1, nn.functional.mse_loss(fsdp_fp32.forward(1, x1), y1))
    fsdp_fp32.sync_gradients()
    fp32_norm = fsdp_fp32.clip_grad_norm(max_norm=10.0)[0]

    for precision, expected_dtype in [("bfloat16", torch.bfloat16), ("float16", torch.float16)]:
        fsdp_amp = VirtualFSDP(copy.deepcopy(base_model), grid=grid, amp_dtype=precision)
        assert fsdp_amp.amp_dtype == expected_dtype
        optimizers = fsdp_amp.create_optimizers(torch.optim.SGD, lr=0.05)

        out0 = fsdp_amp.forward(0, x0)
        out1 = fsdp_amp.forward(1, x1)
        assert out0.dtype == expected_dtype
        assert out1.dtype == expected_dtype

        eval_out = fsdp_amp.eval_forward(0, x0)
        assert eval_out.dtype == expected_dtype
        assert eval_out.requires_grad is False

        loss0 = nn.functional.mse_loss(out0, y0)
        loss1 = nn.functional.mse_loss(out1, y1)
        fsdp_amp.backward(0, loss0)
        fsdp_amp.backward(1, loss1)
        fsdp_amp.sync_gradients()

        amp_norms = fsdp_amp.clip_grad_norm(max_norm=10.0)
        assert abs(amp_norms[0] - fp32_norm) < 0.05

        fsdp_amp.step(optimizers, max_grad_norm=10.0)
        full_params = fsdp_amp.get_full_parameters()
        for orig_name, orig_p in base_model.named_parameters():
            assert not torch.isnan(full_params[orig_name]).any()
            assert not torch.allclose(full_params[orig_name], orig_p.data)

    # Verify FP16 overflow skips parameter update and backs off GradScaler scale factor
    fsdp_fp16 = VirtualFSDP(copy.deepcopy(base_model), grid=grid, amp_dtype="float16")
    opts_fp16 = fsdp_fp16.create_optimizers(torch.optim.SGD, lr=0.1)
    initial_scale = fsdp_fp16.get_grad_scale(0)
    weights_before = fsdp_fp16.get_full_parameters()

    fsdp_fp16.backward(0, nn.functional.mse_loss(fsdp_fp16.forward(0, x0), y0))
    fsdp_fp16.backward(1, nn.functional.mse_loss(fsdp_fp16.forward(1, x1), y1))
    p0 = dict(fsdp_fp16._exec_replicas[0].named_parameters())["fc.weight"]
    assert p0.grad is not None
    p0.grad[0, 0] = float("inf")

    fsdp_fp16.sync_gradients()
    fsdp_fp16.step(opts_fp16)

    assert fsdp_fp16.get_grad_scale(0) == initial_scale * 0.5
    weights_after = fsdp_fp16.get_full_parameters()
    for name in weights_before:
        assert torch.allclose(weights_after[name], weights_before[name])


def test_virtual_fsdp_clip_grad_norm_before_sync_raises() -> None:
    """Verify clip_grad_norm before sync_gradients raises RuntimeError to protect unscale state."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=4.0, device_type="cpu", seed=42)
    model = SimpleLinear()
    fsdp = VirtualFSDP(model, grid=grid)

    x0 = torch.randn(2, 8)
    y0 = torch.zeros(2, 4)
    fsdp.backward(0, nn.functional.mse_loss(fsdp.forward(0, x0), y0))

    with pytest.raises(
        RuntimeError,
        match="VirtualFSDP.clip_grad_norm\\(\\) cannot be called before sync_gradients",
    ):
        fsdp.clip_grad_norm(max_norm=1.0)

    optimizers = fsdp.create_optimizers(torch.optim.SGD, lr=0.01)
    with pytest.raises(
        RuntimeError,
        match="VirtualFSDP.step\\(\\) cannot be called before sync_gradients",
    ):
        fsdp.step(optimizers)


def test_virtual_fsdp_clip_grad_norm_infinite_and_validations() -> None:
    """Verify infinite norms (+inf, -inf, string 'inf') and validation checks."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=4.0, device_type="cpu", seed=42)
    model = SimpleLinear()
    fsdp = VirtualFSDP(model, grid=grid)

    with pytest.raises(ValueError, match="max_norm must be a non-negative finite float"):
        fsdp.clip_grad_norm(max_norm=-1.0)

    with pytest.raises(ValueError, match="norm_type must be greater than 0"):
        fsdp.clip_grad_norm(max_norm=1.0, norm_type=0.0)

    x0 = torch.full((2, 8), 3.0)
    y0 = torch.zeros((2, 4))
    x1 = torch.full((2, 8), 5.0)
    y1 = torch.zeros((2, 4))

    fsdp.backward(0, nn.functional.mse_loss(fsdp.forward(0, x0), y0))
    fsdp.backward(1, nn.functional.mse_loss(fsdp.forward(1, x1), y1))
    fsdp.sync_gradients()

    # Test string 'inf' and float('inf')
    norm_str_inf = fsdp.clip_grad_norm(max_norm=100.0, norm_type="inf")
    norm_float_inf = fsdp.clip_grad_norm(max_norm=100.0, norm_type=float("inf"))
    assert abs(norm_str_inf[0] - norm_float_inf[0]) < 1e-6
    assert norm_str_inf[0] > 0.0

    # Test negative infinity norm
    norm_neg_inf = fsdp.clip_grad_norm(max_norm=100.0, norm_type=float("-inf"))
    assert norm_neg_inf[0] <= norm_float_inf[0]


def test_virtual_fsdp_enable_grad_scaler_and_assert_parameters_synced() -> None:
    """Verify enable_grad_scaler flag and assert_parameters_synced symmetry."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=4.0, device_type="cpu", seed=42)
    model = SimpleLinear()
    fsdp = VirtualFSDP(model, grid=grid, amp_dtype="float16", enable_grad_scaler=False)
    assert fsdp._amp.scaler_enabled is False
    assert fsdp.assert_parameters_synced() is True


def test_virtual_fsdp_gradient_presence_mismatch_raises() -> None:
    """Verify sync_gradients raises RuntimeError if gradients exist on some ranks but not all."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=4.0, device_type="cpu", seed=42)
    model = SimpleLinear()
    fsdp = VirtualFSDP(model, grid=grid)

    x = torch.randn(2, 8)
    out0 = fsdp.forward(0, x)
    out0.sum().backward()
    # Rank 1 computes NO gradient

    with pytest.raises(RuntimeError, match="Gradient presence mismatch for parameter 'fc.weight'"):
        fsdp.sync_gradients()

    # Verify unscaled state was NOT prematurely set
    assert fsdp._amp._grads_unscaled is False


def test_virtual_fsdp_padded_shard_negative_inf_norm_matches_ddp() -> None:
    """Verify negative infinity norm ignores shard padding zeros and matches DDP global norm."""
    from amd_spare_bedroom import VirtualDDP

    grid = VirtualGrid.create(world_size=2, total_memory_gb=4.0, device_type="cpu", seed=42)

    # 5 parameters across 2 ranks -> shard_len = 3, padded_len = 6, padding_len = 1 on rank 1
    model = nn.Linear(5, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))

    fsdp = VirtualFSDP(model, grid=grid)
    ddp = VirtualDDP(model, grid=grid)

    x0 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
    x1 = torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0]])
    y = torch.tensor([[0.0]])

    # Backward on FSDP
    loss0_f = nn.functional.mse_loss(fsdp.forward(0, x0), y)
    loss1_f = nn.functional.mse_loss(fsdp.forward(1, x1), y)
    fsdp.backward(0, loss0_f)
    fsdp.backward(1, loss1_f)
    fsdp.sync_gradients()

    # Backward on DDP
    loss0_d = nn.functional.mse_loss(ddp.forward(0, x0), y)
    loss1_d = nn.functional.mse_loss(ddp.forward(1, x1), y)
    ddp.backward(0, loss0_d)
    ddp.backward(1, loss1_d)
    ddp.sync_gradients()

    fsdp_neg_inf = fsdp.clip_grad_norm(max_norm=100.0, norm_type=float("-inf"))[0]
    ddp_neg_inf = ddp.clip_grad_norm(max_norm=100.0, norm_type=float("-inf"))[0]

    # Verify that FSDP negative inf norm does NOT collapse to 0.0 due to padding zeros!
    assert fsdp_neg_inf > 0.0
    assert math.isclose(fsdp_neg_inf, ddp_neg_inf, rel_tol=1e-5)


def test_virtual_fsdp_eval_forward_while_training_preserves_state() -> None:
    """Verify eval_forward disables dropout during execution and restores training mode."""
    grid = VirtualGrid.create(world_size=2, total_memory_gb=4.0, device_type="cpu", seed=42)

    class DropoutLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.drop = nn.Dropout(p=0.999)
            self.fc = nn.Linear(8, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.fc(self.drop(x)))

    model = DropoutLinear()
    fsdp = VirtualFSDP(model, grid=grid)
    fsdp.train()
    assert fsdp.is_training is True

    x = torch.ones(2, 8)
    out1 = fsdp.eval_forward(0, x)
    out2 = fsdp.eval_forward(0, x)
    assert torch.allclose(out1, out2)

    assert fsdp.is_training is True
    assert fsdp._exec_replicas[0].training is True
    assert fsdp._exec_replicas[0].drop.training is True
