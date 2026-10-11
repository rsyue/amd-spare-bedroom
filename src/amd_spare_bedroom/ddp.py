"""VirtualDDP: Distributed Data Parallel emulation across virtual GPU ranks."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

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
    ) -> None:
        self.grid = grid
        self.communicator = communicator if communicator is not None else VirtualCommunicator(grid)

        # Allocate an isolated model replica for each virtual rank
        self.replicas: dict[int, ModelT] = {rank: copy.deepcopy(module) for rank in self.grid.ranks}

    def get_replica(self, rank: int) -> ModelT:
        """Retrieve the model replica for a specific virtual rank."""
        if rank not in self.replicas:
            raise IndexError(f"Rank {rank} not found in grid ranks {self.grid.ranks}")
        return self.replicas[rank]

    def forward(self, rank: int, *args: Any, **kwargs: Any) -> Any:
        """Execute a forward pass on a specific virtual rank's model replica."""
        return self.get_replica(rank)(*args, **kwargs)

    def sync_gradients(self) -> None:
        """Perform All-Reduce gradient synchronization across all virtual rank replicas.

        Averages gradients computed locally on each rank and updates each replica's
        grad tensor with the synchronized average.
        """
        first_replica = self.replicas[self.grid.ranks[0]]
        param_names = [name for name, _ in first_replica.named_parameters()]

        for name in param_names:
            # Check if gradient exists on rank 0
            p0 = dict(first_replica.named_parameters())[name]
            if p0.grad is None:
                continue

            # Gather gradients across all virtual ranks
            rank_grads: dict[int, torch.Tensor] = {}
            for rank in self.grid.ranks:
                p_rank = dict(self.replicas[rank].named_parameters())[name]
                if p_rank.grad is None:
                    raise RuntimeError(f"Gradient missing on rank {rank} for parameter '{name}'")
                rank_grads[rank] = p_rank.grad

            # Execute All-Reduce mean reduction
            reduced_grads = self.communicator.all_reduce(rank_grads, op="mean")

            # Copy synchronized gradients back to each rank's parameters
            for rank in self.grid.ranks:
                p_rank = dict(self.replicas[rank].named_parameters())[name]
                assert p_rank.grad is not None
                p_rank.grad.copy_(reduced_grads[rank])

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

    def step(self, optimizers: dict[int, torch.optim.Optimizer]) -> None:
        """Execute an optimizer step and zero gradients across all replicas in lockstep."""
        for rank in self.grid.ranks:
            optimizers[rank].step()
            optimizers[rank].zero_grad()
        self.communicator.barrier()

    def assert_parameters_synced(self, atol: float = 1e-6) -> bool:
        """Assert that all rank replicas hold identical parameter values."""
        ref_params = list(self.replicas[self.grid.ranks[0]].parameters())

        for rank in self.grid.ranks[1:]:
            replica_params = list(self.replicas[rank].parameters())
            for p_ref, p_rep in zip(ref_params, replica_params, strict=True):
                if not torch.allclose(p_ref, p_rep, atol=atol):
                    return False
        return True
