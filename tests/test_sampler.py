"""Tests for Seam 3: VirtualRankSampler dataset partitioning and rank isolation."""

import pytest

from amd_spare_bedroom.sampler import VirtualRankSampler


def test_sampler_non_overlapping_partitions() -> None:
    """Verify that samplers across all ranks yield mutually disjoint index sets in an epoch."""
    dataset_size = 100
    num_replicas = 4

    samplers = [
        VirtualRankSampler(
            dataset_size=dataset_size,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            seed=42,
        )
        for rank in range(num_replicas)
    ]

    # Collect indices for each rank
    rank_indices = [list(sampler) for sampler in samplers]

    # Check each rank receives the expected number of samples (equal size)
    for indices in rank_indices:
        assert len(indices) == dataset_size // num_replicas

    # Verify mutual exclusivity: intersection between any two distinct ranks must be empty
    for i in range(num_replicas):
        for j in range(i + 1, num_replicas):
            intersection = set(rank_indices[i]).intersection(set(rank_indices[j]))
            assert len(intersection) == 0, (
                f"Data leakage between rank {i} and rank {j}: {intersection}"
            )

    # Verify all dataset indices are covered
    all_seen = set().union(*[set(indices) for indices in rank_indices])
    assert all_seen == set(range(dataset_size))


def test_sampler_deterministic_reproducibility() -> None:
    """Verify identical epoch and seed yield identical sample sequences."""
    dataset_size = 50
    s1 = VirtualRankSampler(dataset_size=dataset_size, num_replicas=2, rank=0, seed=123)
    s2 = VirtualRankSampler(dataset_size=dataset_size, num_replicas=2, rank=0, seed=123)

    assert list(s1) == list(s2)

    # Change epoch on both
    s1.set_epoch(1)
    s2.set_epoch(1)
    assert list(s1) == list(s2)

    # Different epochs produce different permutations
    s2.set_epoch(2)
    assert list(s1) != list(s2)


def test_sampler_sequential_no_shuffle() -> None:
    """Verify sequential striding when shuffle=False."""
    dataset_size = 10
    num_replicas = 2

    s0 = VirtualRankSampler(
        dataset_size=dataset_size, num_replicas=num_replicas, rank=0, shuffle=False
    )
    s1 = VirtualRankSampler(
        dataset_size=dataset_size, num_replicas=num_replicas, rank=1, shuffle=False
    )

    assert list(s0) == [0, 2, 4, 6, 8]
    assert list(s1) == [1, 3, 5, 7, 9]


def test_sampler_invalid_rank() -> None:
    """Verify invalid rank raises ValueError."""
    with pytest.raises(ValueError, match="Invalid rank"):
        VirtualRankSampler(dataset_size=10, num_replicas=2, rank=2)

    with pytest.raises(ValueError, match="Invalid rank"):
        VirtualRankSampler(dataset_size=10, num_replicas=2, rank=-1)
