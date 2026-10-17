"""VirtualDDP: Distributed Data Parallel emulation across virtual GPU ranks."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import torch
import torch.nn as nn

from amd_spare_bedroom.amp import ROCM_GRAD_FINITE_BOUND, AmpGradScalerState
from amd_spare_bedroom.communicator import VirtualCommunicator

if TYPE_CHECKING:
    from amd_spare_bedroom.grid import VirtualGrid


class VirtualDDP[ModelT: nn.Module]:
    """Emulates Distributed Data Parallel (DDP) across virtual device ranks.

    Maintains rank-isolated model replicas with address-space memory quotas,
    synchronizes backward gradients via collective All-Reduce, and ensures
    lockstep weight progression across virtual ranks.
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
        self._is_training: bool = module.training
        self._amp = AmpGradScalerState(
            device_type=self.grid.device_type,
            ranks=self.grid.ranks,
            amp_dtype=amp_dtype,
            enable_grad_scaler=enable_grad_scaler,
        )
        self.amp_dtype: torch.dtype | None = self._amp.amp_dtype

        target_device = torch.device(self.grid.device_type)
        # Allocate an isolated model replica for each virtual rank
        self.replicas: dict[int, ModelT] = {
            rank: copy.deepcopy(module).to(target_device) for rank in self.grid.ranks
        }
        self._gradients_synced: bool = False

    @property
    def is_training(self) -> bool:
        """Return whether the distributed wrapper is currently in training mode."""
        return self._is_training

    def get_grad_scale(self, rank: int = 0) -> float:
        """Return the active AMP GradScaler scale factor for the given rank."""
        return self._amp.get_scale(rank)

    def train(self, mode: bool = True) -> Self:
        """Set training mode across all rank model replicas."""
        self._is_training = mode
        for replica in self.replicas.values():
            replica.train(mode)
        return self

    def eval(self) -> Self:
        """Set evaluation mode across all rank model replicas."""
        return self.train(False)

    def get_replica(self, rank: int) -> ModelT:
        """Retrieve the model replica for a specific virtual rank."""
        if rank not in self.replicas:
            raise IndexError(f"Rank {rank} not found in grid ranks {self.grid.ranks}")
        return self.replicas[rank]

    def unwrap_model(self) -> ModelT:
        """Return the synchronized un-sharded model replica on rank 0."""
        return self.replicas[self.grid.ranks[0]]

    def state_dict(self) -> dict[str, Any]:
        """Return the full model state_dict from rank 0."""
        return self.unwrap_model().state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load state_dict into all rank replicas in lockstep."""
        for replica in self.replicas.values():
            replica.load_state_dict(state_dict)

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        safe_serialization: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save the unwrapped HuggingFace model to save_directory in safetensors format."""
        raw_model: Any = self.unwrap_model()
        raw_model.save_pretrained(
            save_directory,
            safe_serialization=safe_serialization,
            **kwargs,
        )

    def forward(self, rank: int, *args: Any, **kwargs: Any) -> Any:
        """Execute a forward pass on a specific virtual rank's model replica."""
        replica = self.get_replica(rank)
        with self._amp.autocast_context():
            return replica(*args, **kwargs)

    def eval_forward(self, rank: int, *args: Any, **kwargs: Any) -> Any:
        """Execute an evaluation forward pass on a virtual rank under torch.no_grad()."""
        replica = self.get_replica(rank)
        prev_states = {m: m.training for m in replica.modules()}
        replica.eval()
        try:
            with torch.no_grad():
                return self.forward(rank, *args, **kwargs)
        finally:
            for m, state in prev_states.items():
                m.train(state)

    def backward(self, rank: int, loss: torch.Tensor) -> None:
        """Execute a backward pass on a virtual rank, scaling loss when FP16 GradScaler is active."""
        if rank not in self.replicas:
            raise IndexError(f"Rank {rank} not found in grid ranks {self.grid.ranks}")
        self._gradients_synced = False
        self._amp.backward(rank, loss)

    def sync_gradients(self) -> None:
        """Perform All-Reduce gradient synchronization across all virtual rank replicas.

        Unscales FP16 gradients (detecting overflow prior to sanitization), averages
        gradients computed locally on each rank, and updates each replica's grad tensor.
        """
        first_replica = self.replicas[self.grid.ranks[0]]
        param_names = [name for name, _ in first_replica.named_parameters()]
        rank_params = {
            rank: dict(self.replicas[rank].named_parameters()) for rank in self.grid.ranks
        }

        for name in param_names:
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

        self._amp.unscale_and_check_finite(
            {rank: self.replicas[rank].parameters() for rank in self.grid.ranks}
        )

        for name in param_names:
            grads_present = [rank_params[r][name].grad is not None for r in self.grid.ranks]
            if not any(grads_present):
                continue

            # Gather gradients across all virtual ranks, sanitizing non-finite values (NaN/Inf)
            # to prevent ROCm kernel underflow/overflow artifacts from poisoning the reduction
            rank_grads: dict[int, torch.Tensor] = {}
            for rank in self.grid.ranks:
                p_rank = rank_params[rank][name]
                assert p_rank.grad is not None
                p_rank.grad.nan_to_num_(
                    nan=0.0, posinf=ROCM_GRAD_FINITE_BOUND, neginf=-ROCM_GRAD_FINITE_BOUND
                )
                rank_grads[rank] = p_rank.grad

            # Execute All-Reduce mean reduction
            reduced_grads = self.communicator.all_reduce(rank_grads, op="mean")

            # Copy synchronized gradients back to each rank's parameters
            for rank in self.grid.ranks:
                p_rank = rank_params[rank][name]
                assert p_rank.grad is not None
                p_rank.grad.copy_(reduced_grads[rank])

        self._gradients_synced = True

    def clip_grad_norm(
        self,
        max_norm: float,
        norm_type: float | str = 2.0,
    ) -> dict[int, float]:
        """Clip gradient norms across all virtual rank replicas in lockstep.

        Returns:
            Mapping from rank ID to the pre-clipping total gradient norm on that rank.
        """
        if max_norm is None or not math.isfinite(max_norm) or max_norm < 0.0:
            raise ValueError(f"max_norm must be a non-negative finite float, got {max_norm}")

        try:
            norm_type_f = float(norm_type)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"norm_type must be a valid float or float string, got {norm_type!r}"
            ) from e

        if not math.isinf(norm_type_f) and norm_type_f <= 0:
            raise ValueError(f"norm_type must be greater than 0, got {norm_type_f}")

        if not self._gradients_synced:
            raise RuntimeError(
                "VirtualDDP.clip_grad_norm() cannot be called before sync_gradients(). "
                "Gradients must be synchronized across virtual ranks before clipping."
            )

        self._amp.unscale_and_check_finite(
            {rank: self.replicas[rank].parameters() for rank in self.grid.ranks}
        )
        norms: dict[int, float] = {}
        for rank in self.grid.ranks:
            replica_params = [p for p in self.replicas[rank].parameters() if p.grad is not None]
            for p in replica_params:
                assert p.grad is not None
                p.grad.nan_to_num_(
                    nan=0.0, posinf=ROCM_GRAD_FINITE_BOUND, neginf=-ROCM_GRAD_FINITE_BOUND
                )
            if not replica_params:
                norms[rank] = 0.0
                continue
            total_norm = torch.nn.utils.clip_grad_norm_(
                replica_params,
                max_norm=max_norm,
                norm_type=norm_type_f,
            )
            norms[rank] = float(total_norm.item())
        self.communicator.barrier()
        return norms

    def create_optimizers(
        self,
        optimizer_cls: type[torch.optim.Optimizer],
        **optimizer_kwargs: Any,
    ) -> dict[int, torch.optim.Optimizer]:
        """Instantiate an optimizer per rank replica."""
        return {
            rank: optimizer_cls(self.replicas[rank].parameters(), **optimizer_kwargs)
            for rank in self.grid.ranks
        }

    def step(
        self,
        optimizers: dict[int, torch.optim.Optimizer],
        max_grad_norm: float | None = None,
    ) -> dict[int, float] | None:
        """Execute an optimizer step and zero gradients across all replicas in lockstep."""
        if not self._gradients_synced:
            raise RuntimeError(
                "VirtualDDP.step() cannot be called before sync_gradients(). "
                "Gradients must be synchronized across virtual ranks before stepping optimizers."
            )

        norms: dict[int, float] | None = None
        if max_grad_norm is not None and max_grad_norm > 0:
            norms = self.clip_grad_norm(max_grad_norm)
        self._amp.step_optimizers(
            optimizers,
            {rank: self.replicas[rank].parameters() for rank in self.grid.ranks},
        )
        self.communicator.barrier()
        self._gradients_synced = False
        return norms

    def get_full_parameters(self) -> dict[str, torch.Tensor]:
        """Retrieve the full parameter dictionary (from rank 0 replica)."""
        first_replica = self.replicas[self.grid.ranks[0]]
        return {name: p.data.clone() for name, p in first_replica.named_parameters()}

    def assert_parameters_synced(self, atol: float = 1e-6) -> bool:
        """Assert that all rank replicas hold identical parameter values."""
        ref_params = list(self.replicas[self.grid.ranks[0]].parameters())

        for rank in self.grid.ranks[1:]:
            replica_params = list(self.replicas[rank].parameters())
            for p_ref, p_rep in zip(ref_params, replica_params, strict=True):
                if not torch.allclose(p_ref, p_rep, atol=atol):
                    return False
        return True
