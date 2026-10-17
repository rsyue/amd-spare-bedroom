"""VirtualRankSampler for deterministic, non-overlapping dataset partitioning."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch


class VirtualRankSampler:
    """Dataset sampler distributing non-overlapping data slices to a Virtual Rank.

    Mirrors PyTorch's DistributedSampler semantics, ensuring zero data leakage
    across virtual devices and deterministic, seedable shuffling per epoch.
    """

    def __init__(
        self,
        dataset_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError(f"dataset_size must be positive, got {dataset_size}")
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank {rank} for num_replicas {num_replicas}")

        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch: int = 0

        if self.drop_last and self.dataset_size % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (self.dataset_size - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(self.dataset_size / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number for deterministic pseudo-random shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """Yield index partition for this rank."""
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=g).tolist()
        else:
            indices = list(range(self.dataset_size))

        if not self.drop_last:
            # Add padding to make dataset size evenly divisible by num_replicas
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Drop trailing elements
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size

        # Subsample for this rank via striding
        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(rank_indices) == self.num_samples

        return iter(rank_indices)

    def __len__(self) -> int:
        """Return the number of samples for this rank."""
        return self.num_samples

    @staticmethod
    def unstride_and_unpad(
        rank_tensors: Sequence[torch.Tensor],
        dataset_size: int,
    ) -> torch.Tensor:
        """Reconstruct sequential dataset order from stride-partitioned rank outputs and strip padding.

        Args:
            rank_tensors: Per-rank tensors of identical shape and dtype produced by VirtualRankSampler.
            dataset_size: Total unpadded sample count to retain.

        Returns:
            Contiguous interleaved tensor sliced to dataset_size.

        Raises:
            ValueError: If rank_tensors is empty, dataset_size <= 0, any tensor has ndim == 0,
                tensor shapes mismatch, or gathered samples are fewer than dataset_size.
        """
        tensor_list = list(rank_tensors)
        if not tensor_list:
            raise ValueError("rank_tensors must be a non-empty sequence of tensors.")
        if dataset_size <= 0:
            raise ValueError(f"dataset_size must be a positive integer, got {dataset_size}.")
        if any(t.ndim == 0 for t in tensor_list):
            raise ValueError("Tensors in rank_tensors must have ndim >= 1.")

        first_shape = tensor_list[0].shape
        for i, t in enumerate(tensor_list):
            if t.shape != first_shape:
                raise ValueError(
                    f"Rank tensor shape mismatch at index {i}: expected {first_shape}, got {t.shape}."
                )

        stacked = torch.stack(tensor_list, dim=0).transpose(0, 1)
        interleaved = stacked.reshape(-1, *stacked.shape[2:])
        if interleaved.shape[0] < dataset_size:
            raise ValueError(
                f"Gathered elements ({interleaved.shape[0]}) are fewer than requested dataset_size ({dataset_size})."
            )
        return interleaved[:dataset_size].contiguous()
