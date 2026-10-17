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


def test_sampler_unstride_and_unpad_success() -> None:
    """Verify unstride_and_unpad correctly reverses striding and strips padding for 1D and 2D tensors."""
    import torch

    # 1D labels: original dataset was 5 samples: [10, 20, 30, 40, 50]
    # Strided across 2 ranks with padding to 6:
    # rank 0 gets [10, 30, 50]
    # rank 1 gets [20, 40, 10] (padded with index 0 -> 10)
    r0 = torch.tensor([10, 30, 50])
    r1 = torch.tensor([20, 40, 10])

    unpadded = VirtualRankSampler.unstride_and_unpad([r0, r1], dataset_size=5)
    assert torch.equal(unpadded, torch.tensor([10, 20, 30, 40, 50]))
    assert unpadded.is_contiguous()

    # 2D tensors (e.g. logits of shape [N, 2]): 3 ranks, 7 total samples, padded to 9 (3 per rank)
    r0_2d = torch.tensor([[1.0, 0.0], [4.0, 0.0], [7.0, 0.0]])
    r1_2d = torch.tensor([[2.0, 0.0], [5.0, 0.0], [1.0, 0.0]])  # padded
    r2_2d = torch.tensor([[3.0, 0.0], [6.0, 0.0], [2.0, 0.0]])  # padded

    unpadded_2d = VirtualRankSampler.unstride_and_unpad([r0_2d, r1_2d, r2_2d], dataset_size=7)
    expected_2d = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
            [7.0, 0.0],
        ]
    )
    assert torch.equal(unpadded_2d, expected_2d)

    # Single-rank identity case (world_size = 1)
    single_rank = torch.tensor([100, 200, 300])
    assert torch.equal(
        VirtualRankSampler.unstride_and_unpad([single_rank], dataset_size=3), single_rank
    )


def test_sampler_unstride_and_unpad_defensive_errors() -> None:
    """Verify defensive exceptions in unstride_and_unpad."""
    import torch

    r0 = torch.tensor([1, 2])
    r1 = torch.tensor([3, 4])

    with pytest.raises(ValueError, match="rank_tensors must be a non-empty sequence"):
        VirtualRankSampler.unstride_and_unpad([], dataset_size=4)

    with pytest.raises(ValueError, match="dataset_size must be a positive integer"):
        VirtualRankSampler.unstride_and_unpad([r0, r1], dataset_size=0)

    with pytest.raises(ValueError, match="dataset_size must be a positive integer"):
        VirtualRankSampler.unstride_and_unpad([r0, r1], dataset_size=-5)

    with pytest.raises(ValueError, match="Tensors in rank_tensors must have ndim >= 1"):
        VirtualRankSampler.unstride_and_unpad([torch.tensor(1), torch.tensor(2)], dataset_size=2)

    with pytest.raises(ValueError, match="Rank tensor shape mismatch"):
        VirtualRankSampler.unstride_and_unpad([r0, torch.tensor([1, 2, 3])], dataset_size=2)

    with pytest.raises(ValueError, match="Gathered elements .* are fewer than requested"):
        VirtualRankSampler.unstride_and_unpad([r0, r1], dataset_size=10)
