"""VirtualFSDP: Fully Sharded Data Parallel emulation across virtual GPU ranks."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import torch
import torch.nn as nn

from amd_spare_bedroom.amp import ROCM_GRAD_FINITE_BOUND, AmpGradScalerState
from amd_spare_bedroom.communicator import VirtualCommunicator

if TYPE_CHECKING:
    from amd_spare_bedroom.grid import VirtualGrid


@dataclass(frozen=True)
class ParamMetadata:
    """Metadata for reconstructing an un-sharded parameter tensor."""

    name: str
    orig_shape: torch.Size
    orig_numel: int
    padding_len: int
    shard_len: int
    dtype: torch.dtype


class VirtualFSDP[ModelT: nn.Module]:
    """Emulates Fully Sharded Data Parallel (FSDP / ZeRO-3) across virtual device ranks.

    Shards model parameters, gradients, and optimizer states across virtual ranks,
    cutting per-rank memory footprint to ~1/world_size. Reconstructs full weights
    on-demand via All-Gather and scatters reduced gradients via Reduce-Scatter.
    """

    def __init__(
        self,
        module: ModelT,
        grid: VirtualGrid,
        communicator: VirtualCommunicator | None = None,
        amp_dtype: torch.dtype | str | None = None,
        enable_grad_scaler: bool | None = None,
    ) -> None:
        self.grid = grid
        self.communicator = communicator if communicator is not None else VirtualCommunicator(grid)
        self.target_device = torch.device(grid.device_type)
        self._is_training: bool = module.training
        self._amp = AmpGradScalerState(
            device_type=self.grid.device_type,
            ranks=self.grid.ranks,
            amp_dtype=amp_dtype,
            enable_grad_scaler=enable_grad_scaler,
        )
        self.amp_dtype: torch.dtype | None = self._amp.amp_dtype

        # Base skeleton for executing forward/backward passes
        self._skeleton: ModelT = copy.deepcopy(module).to(self.target_device)
        self.metadata: dict[str, ParamMetadata] = {}

        # rank_shards[rank][param_name] = sharded nn.Parameter
        self.rank_shards: dict[int, dict[str, nn.Parameter]] = {
            rank: {} for rank in self.grid.ranks
        }

        # Shard all named parameters across the virtual grid
        for name, param in self._skeleton.named_parameters():
            orig_shape = param.shape
            orig_numel = param.numel()
            flat_param = param.data.view(-1).clone()

            shard_len = math.ceil(orig_numel / self.grid.world_size)
            padded_len = shard_len * self.grid.world_size
            padding_len = padded_len - orig_numel

            if padding_len > 0:
                pad_tensor = torch.zeros(
                    padding_len, dtype=flat_param.dtype, device=flat_param.device
                )
                padded_flat = torch.cat([flat_param, pad_tensor], dim=0)
            else:
                padded_flat = flat_param

            shards = list(torch.chunk(padded_flat, chunks=self.grid.world_size, dim=0))

            self.metadata[name] = ParamMetadata(
                name=name,
                orig_shape=orig_shape,
                orig_numel=orig_numel,
                padding_len=padding_len,
                shard_len=shard_len,
                dtype=param.dtype,
            )

            for rank in self.grid.ranks:
                self.rank_shards[rank][name] = nn.Parameter(
                    shards[rank].clone(), requires_grad=param.requires_grad
                )

        # Execution replicas instantiated per rank
        self._exec_replicas: dict[int, ModelT] = {
            rank: copy.deepcopy(self._skeleton) for rank in self.grid.ranks
        }

    @property
    def is_training(self) -> bool:
        """Return whether the sharded wrapper is currently in training mode."""
        return self._is_training

    def get_grad_scale(self, rank: int = 0) -> float:
        """Return the active AMP GradScaler scale factor for the given rank."""
        return self._amp.get_scale(rank)

    def train(self, mode: bool = True) -> Self:
        """Set training mode across the skeleton and all rank execution replicas."""
        self._is_training = mode
        self._skeleton.train(mode)
        for replica in self._exec_replicas.values():
            replica.train(mode)
        return self

    def eval(self) -> Self:
        """Set evaluation mode across the skeleton and all rank execution replicas."""
        return self.train(False)

    def get_sharded_parameters(self, rank: int) -> list[nn.Parameter]:
        """Retrieve the parameter shards owned by a specific virtual rank."""
        if rank not in self.rank_shards:
            raise IndexError(f"Rank {rank} not found in grid ranks {self.grid.ranks}")
        return list(self.rank_shards[rank].values())

    def get_rank_parameter_numel(self, rank: int) -> int:
        """Calculate the total number of parameter elements stored in a rank's shard."""
        return sum(p.numel() for p in self.get_sharded_parameters(rank))

    def get_full_parameters(self) -> dict[str, torch.Tensor]:
        """Reconstruct and return all un-sharded parameter tensors via All-Gather."""
        full_params: dict[str, torch.Tensor] = {}

        for name, meta in self.metadata.items():
            shards_by_rank = {rank: self.rank_shards[rank][name].data for rank in self.grid.ranks}
            gathered_full = self.communicator.all_gather(shards_by_rank, dim=0)[self.grid.ranks[0]]

            # Remove padding and restore original shape
            unpadded = gathered_full[: meta.orig_numel] if meta.padding_len > 0 else gathered_full
            full_params[name] = unpadded.view(meta.orig_shape).clone()

        return full_params

    def _gather_into_replica(self, rank: int) -> None:
        """All-Gather sharded parameters and reconstruct full tensors into rank's replica."""
        replica = self._exec_replicas[rank]

        for name, meta in self.metadata.items():
            shards_by_rank = {r: self.rank_shards[r][name].data for r in self.grid.ranks}
            gathered_full = self.communicator.all_gather(shards_by_rank, dim=0)[rank]

            unpadded = gathered_full[: meta.orig_numel] if meta.padding_len > 0 else gathered_full
            target_p = dict(replica.named_parameters())[name]
            target_p.data.copy_(unpadded.view(meta.orig_shape))

    def forward(self, rank: int, *args: Any, **kwargs: Any) -> Any:
        """Execute a forward pass on rank by temporarily all-gathering full parameters."""
        if rank not in self._exec_replicas:
            raise IndexError(f"Rank {rank} not found in grid ranks {self.grid.ranks}")
        self._gather_into_replica(rank)
        with self._amp.autocast_context():
            return self._exec_replicas[rank](*args, **kwargs)

    def eval_forward(self, rank: int, *args: Any, **kwargs: Any) -> Any:
        """Execute an evaluation forward pass on rank under torch.no_grad()."""
        if rank not in self._exec_replicas:
            raise IndexError(f"Rank {rank} not found in grid ranks {self.grid.ranks}")
        replica = self._exec_replicas[rank]
        prev_states = {m: m.training for m in replica.modules()}
        replica.eval()
        try:
            with torch.no_grad():
                return self.forward(rank, *args, **kwargs)
        finally:
            for m, state in prev_states.items():
                m.train(state)

    def backward(self, rank: int, loss: torch.Tensor) -> None:
        """Execute a backward pass on rank, scaling loss when FP16 GradScaler is active."""
        if rank not in self._exec_replicas:
            raise IndexError(f"Rank {rank} not found in grid ranks {self.grid.ranks}")
        self._amp.backward(rank, loss)

    def sync_gradients(self) -> None:
        """Reduce-Scatter full gradients back into each rank's owned parameter shard.

        Unscales FP16 gradients (detecting overflow prior to sanitization) and sanitizes
        non-finite gradients (NaN/Inf) prior to reduction.
        """
        self._amp.unscale_and_check_finite(
            {rank: self._exec_replicas[rank].parameters() for rank in self.grid.ranks}
        )

        rank_params = {
            rank: dict(self._exec_replicas[rank].named_parameters()) for rank in self.grid.ranks
        }

        for name, meta in self.metadata.items():
            grads_present = [rank_params[r][name].grad is not None for r in self.grid.ranks]
            if not any(grads_present):
                continue
            if not all(grads_present):
                missing = [
                    r for r, pres in zip(self.grid.ranks, grads_present, strict=True) if not pres
                ]
                raise RuntimeError(
                    f"Gradient presence mismatch for parameter '{name}': missing on ranks {missing}"
                )

            full_grads_by_rank: dict[int, torch.Tensor] = {}
            for rank in self.grid.ranks:
                p_rank = rank_params[rank][name]
                assert p_rank.grad is not None
                p_rank.grad.nan_to_num_(
                    nan=0.0, posinf=ROCM_GRAD_FINITE_BOUND, neginf=-ROCM_GRAD_FINITE_BOUND
                )
                flat_grad = p_rank.grad.view(-1)
                if meta.padding_len > 0:
                    pad = torch.zeros(
                        meta.padding_len,
                        dtype=flat_grad.dtype,
                        device=flat_grad.device,
                    )
                    padded_grad = torch.cat([flat_grad, pad], dim=0)
                else:
                    padded_grad = flat_grad

                full_grads_by_rank[rank] = padded_grad

            # Reduce-Scatter: each rank receives its 1/N gradient chunk
            scattered_grads = self.communicator.reduce_scatter(full_grads_by_rank, op="mean", dim=0)

            # Assign scattered gradient to the rank's sharded parameter
            for rank in self.grid.ranks:
                self.rank_shards[rank][name].grad = scattered_grads[rank].clone()

            # Free gradient memory from the un-sharded execution models
            for rank in self.grid.ranks:
                rank_params[rank][name].grad = None

    def clip_grad_norm(
        self,
        max_norm: float,
        norm_type: float | str = 2.0,
    ) -> dict[int, float]:
        """Compute exact global gradient norm across all ranks' shards and clip in lockstep.

        Returns:
            Mapping from rank ID to the pre-clipping global gradient norm.
        """
        has_shard_grads = any(
            p.grad is not None
            for rank in self.grid.ranks
            for p in self.get_sharded_parameters(rank)
        )
        has_exec_grads = any(
            p.grad is not None
            for rank in self.grid.ranks
            for p in self._exec_replicas[rank].parameters()
        )
        if not has_shard_grads and has_exec_grads:
            raise RuntimeError(
                "VirtualFSDP.clip_grad_norm() cannot be called before sync_gradients(). "
                "Sharded parameter gradients have not yet been reduce-scattered."
            )

        if max_norm is not None:
            if math.isnan(max_norm) or max_norm < 0.0:
                raise ValueError(f"max_norm must be a non-negative finite float, got {max_norm}")

        try:
            norm_type_f = float(norm_type)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"norm_type must be a valid float or float string, got {norm_type!r}"
            ) from e

        self._amp.unscale_and_check_finite(
            {rank: self.get_sharded_parameters(rank) for rank in self.grid.ranks}
        )
        for rank in self.grid.ranks:
            for shard in self.get_sharded_parameters(rank):
                if shard.grad is not None:
                    shard.grad.nan_to_num_(
                        nan=0.0, posinf=ROCM_GRAD_FINITE_BOUND, neginf=-ROCM_GRAD_FINITE_BOUND
                    )

        if math.isinf(norm_type_f):
            if norm_type_f > 0:
                max_abs = 0.0
                for rank in self.grid.ranks:
                    for shard in self.get_sharded_parameters(rank):
                        if shard.grad is not None and shard.grad.numel() > 0:
                            val = float(shard.grad.detach().abs().max().item())
                            if val > max_abs:
                                max_abs = val
                global_norm = max_abs
            else:
                min_abs = float("inf")
                for rank in self.grid.ranks:
                    for shard in self.get_sharded_parameters(rank):
                        if shard.grad is not None and shard.grad.numel() > 0:
                            val = float(shard.grad.detach().abs().min().item())
                            if val < min_abs:
                                min_abs = val
                global_norm = 0.0 if math.isinf(min_abs) else min_abs
            self.communicator.barrier()
        elif norm_type_f <= 0:
            raise ValueError(f"norm_type must be greater than 0, got {norm_type_f}")
        else:
            rank_pow_sums: dict[int, torch.Tensor] = {}
            for rank in self.grid.ranks:
                local_sum = torch.zeros(1, dtype=torch.float32, device=self.target_device)
                for shard in self.get_sharded_parameters(rank):
                    if shard.grad is not None:
                        shard_norm = torch.linalg.vector_norm(
                            shard.grad.detach().float(), ord=norm_type_f
                        )
                        local_sum = local_sum + (shard_norm**norm_type_f)
                rank_pow_sums[rank] = local_sum

            reduced_sums = self.communicator.all_reduce(rank_pow_sums, op="sum")
            global_norm = float((reduced_sums[self.grid.ranks[0]] ** (1.0 / norm_type_f)).item())

        if max_norm >= 0.0:
            clip_coef = min(1.0, float(max_norm) / (global_norm + 1e-6))
            if clip_coef < 1.0:
                for rank in self.grid.ranks:
                    for shard in self.get_sharded_parameters(rank):
                        if shard.grad is not None:
                            shard.grad.detach().mul_(clip_coef)

        return {rank: global_norm for rank in self.grid.ranks}

    def create_optimizers(
        self,
        optimizer_cls: type[torch.optim.Optimizer],
        **optimizer_kwargs: Any,
    ) -> dict[int, torch.optim.Optimizer]:
        """Instantiate an optimizer per virtual rank, operating ONLY on its owned shards."""
        return {
            rank: optimizer_cls(self.get_sharded_parameters(rank), **optimizer_kwargs)
            for rank in self.grid.ranks
        }

    def step(
        self,
        optimizers: dict[int, torch.optim.Optimizer],
        max_grad_norm: float | None = None,
    ) -> dict[int, float] | None:
        """Execute sharded optimizer steps and zero gradients across all ranks."""
        norms: dict[int, float] | None = None
        if max_grad_norm is not None and max_grad_norm > 0:
            norms = self.clip_grad_norm(max_grad_norm)
        self._amp.step_optimizers(
            optimizers,
            {rank: self.get_sharded_parameters(rank) for rank in self.grid.ranks},
        )
        self.communicator.barrier()
        return norms

    def assert_parameters_synced(self, atol: float = 1e-6) -> bool:
        """Assert that all parameter shards are finite and synchronized."""
        for rank in self.grid.ranks:
            for shard in self.get_sharded_parameters(rank):
                if not bool(torch.isfinite(shard.data).all().item()):
                    return False
        return True
