"""Tests for hyperparameter grid expansion and sequential wave scheduling."""

from __future__ import annotations

import pytest

from amd_spare_bedroom.grid import VirtualGrid
from amd_spare_bedroom.grid_search import (
    HyperparameterGrid,
    TrialConfig,
    WaveSchedule,
)


def _make_5x5_grid() -> HyperparameterGrid:
    return HyperparameterGrid(
        {
            "lr": [1e-5, 2e-5, 3e-5, 5e-5, 1e-4],
            "weight_decay": [0.0, 1e-4, 1e-3, 1e-2, 1e-1],
        }
    )


def test_hyperparameter_grid_5x5_cartesian_expansion() -> None:
    """5 learning rates x 5 weight decays expands to 25 unique TrialConfigs."""
    grid = _make_5x5_grid()

    assert grid.total_trials == 25
    assert len(grid.trials) == 25
    assert [t.trial_id for t in grid.trials] == list(range(25))

    pairs = [(t.params["lr"], t.params["weight_decay"]) for t in grid.trials]
    assert len(set(pairs)) == 25
    assert pairs[0] == (1e-5, 0.0)
    assert pairs[1] == (1e-5, 1e-4)
    assert pairs[-1] == (1e-4, 1e-1)
    assert isinstance(grid.trials[0], TrialConfig)


def test_plan_waves_single_wave_when_world_size_equals_trials() -> None:
    """When world_size == 25 for 25 trials, all run in 1 wave with no idle ranks."""
    grid = _make_5x5_grid()
    waves = grid.plan_waves(world_size=25)

    assert len(waves) == 1
    wave0 = waves[0]
    assert isinstance(wave0, WaveSchedule)
    assert wave0.wave_index == 0
    assert wave0.active_ranks == tuple(range(25))
    assert wave0.idle_ranks == ()
    assert list(wave0.rank_to_trial.keys()) == list(range(25))
    assert [t.trial_id for t in wave0.rank_to_trial.values()] == list(range(25))


def test_plan_waves_sequential_waves_when_world_size_smaller_than_trials() -> None:
    """When world_size == 10 for 25 trials, schedules 3 sequential waves (10, 10, 5)."""
    grid = _make_5x5_grid()
    virtual_grid = VirtualGrid(world_size=10, total_memory_bytes=10 * 1024**3, device_type="cpu")
    waves = grid.plan_waves(virtual_grid)

    assert len(waves) == 3

    # Wave 0: trials 0..9 on ranks 0..9
    assert waves[0].wave_index == 0
    assert waves[0].active_ranks == tuple(range(10))
    assert waves[0].idle_ranks == ()
    assert [waves[0].rank_to_trial[r].trial_id for r in range(10)] == list(range(0, 10))

    # Wave 1: trials 10..19 on ranks 0..9
    assert waves[1].wave_index == 1
    assert waves[1].active_ranks == tuple(range(10))
    assert waves[1].idle_ranks == ()
    assert [waves[1].rank_to_trial[r].trial_id for r in range(10)] == list(range(10, 20))

    # Wave 2: trials 20..24 on active ranks 0..4, idle ranks 5..9
    assert waves[2].wave_index == 2
    assert waves[2].active_ranks == (0, 1, 2, 3, 4)
    assert waves[2].idle_ranks == (5, 6, 7, 8, 9)
    assert [waves[2].rank_to_trial[r].trial_id for r in range(5)] == list(range(20, 25))
    assert set(waves[2].rank_to_trial.keys()) == {0, 1, 2, 3, 4}


def test_plan_waves_excess_ranks_when_world_size_exceeds_trials() -> None:
    """When world_size == 30 for 25 trials, 1 wave has active 0..24 and idle 25..29."""
    grid = _make_5x5_grid()
    waves = grid.plan_waves(world_size=30)

    assert len(waves) == 1
    assert waves[0].wave_index == 0
    assert waves[0].active_ranks == tuple(range(25))
    assert waves[0].idle_ranks == (25, 26, 27, 28, 29)
    assert len(waves[0].rank_to_trial) == 25


def test_hyperparameter_grid_validation_errors() -> None:
    """Reject empty parameter grids, empty value sequences, and invalid world_size."""
    with pytest.raises(ValueError, match="param_grid must not be empty"):
        HyperparameterGrid({})

    with pytest.raises(ValueError, match="must contain at least one candidate value"):
        HyperparameterGrid({"lr": [1e-5], "weight_decay": []})

    grid = HyperparameterGrid({"lr": [1e-5, 2e-5]})
    with pytest.raises(ValueError, match="world_size must be >= 1"):
        grid.plan_waves(0)

    with pytest.raises(ValueError, match="world_size must be >= 1"):
        grid.plan_waves(-2)
