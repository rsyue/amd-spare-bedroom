"""VirtualFSDP: Fully Sharded Data Parallel emulation across virtual GPU ranks."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

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
    ) -> None:
        self.grid = grid
        self.communicator = communicator if communicator is not None else VirtualCommunicator(grid)
        self.target_device = torch.device(grid.device_type)

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
        self._gather_into_replica(rank)
        return self._exec_replicas[rank](*args, **kwargs)

    def sync_gradients(self) -> None:
        """Reduce-Scatter full gradients back into each rank's owned parameter shard."""
        for name, meta in self.metadata.items():
            # Check if gradient exists
            p0 = dict(self._exec_replicas[self.grid.ranks[0]].named_parameters())[name]
            if p0.grad is None:
                continue

            full_grads_by_rank: dict[int, torch.Tensor] = {}
            for rank in self.grid.ranks:
                p_rank = dict(self._exec_replicas[rank].named_parameters())[name]
                if p_rank.grad is None:
                    raise RuntimeError(f"Gradient missing on rank {rank} for parameter '{name}'")

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
                dict(self._exec_replicas[rank].named_parameters())[name].grad = None

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

    def step(self, optimizers: dict[int, torch.optim.Optimizer]) -> None:
        """Execute sharded optimizer steps and zero gradients across all ranks."""
        for rank in self.grid.ranks:
            optimizers[rank].step()
            optimizers[rank].zero_grad()
        self.communicator.barrier()
