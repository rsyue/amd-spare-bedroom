"""Hyperparameter grid expansion and multi-rank sequential wave scheduling.

Provides deterministic Cartesian search space expansion (`HyperparameterGrid`,
`TrialConfig`) and wave partitioning (`WaveSchedule`) across `VirtualGrid` ranks.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from amd_spare_bedroom.grid import VirtualGrid


@dataclass(frozen=True)
class TrialConfig:
    """A single hyperparameter trial configuration in a grid search.

    Attributes:
        trial_id: Zero-indexed sequential identifier (`0 .. total_trials - 1`).
        params: Mapping of hyperparameter names to chosen candidate values.
    """

    trial_id: int
    params: dict[str, Any]


@dataclass(frozen=True)
class WaveSchedule:
    """Assignment of hyperparameter trials to virtual ranks for one execution wave.

    Attributes:
        wave_index: Zero-indexed wave position (`0 .. num_waves - 1`).
        rank_to_trial: Mapping from active virtual rank index (`0 .. K - 1`) to
            its assigned `TrialConfig`.
        active_ranks: Tuple of virtual rank indices executing a trial in this wave.
        idle_ranks: Tuple of virtual rank indices with no assigned trial in this wave.
    """

    wave_index: int
    rank_to_trial: dict[int, TrialConfig]
    active_ranks: tuple[int, ...]
    idle_ranks: tuple[int, ...]


class HyperparameterGrid:
    """Deterministic Cartesian hyperparameter search space and wave scheduler.

    Expands a dictionary of hyperparameter candidate sequences into `TrialConfig`
    objects and partitions them into sequential waves of up to `world_size`
    concurrent virtual ranks.
    """

    def __init__(self, param_grid: Mapping[str, Sequence[Any]]) -> None:
        if not param_grid:
            raise ValueError("param_grid must not be empty.")

        keys: list[str] = []
        value_lists: list[list[Any]] = []
        for key, candidates in param_grid.items():
            if len(candidates) == 0:
                raise ValueError(
                    f"Hyperparameter {key!r} must contain at least one candidate value."
                )
            keys.append(key)
            value_lists.append(list(candidates))

        trials: list[TrialConfig] = []
        for trial_id, combo in enumerate(itertools.product(*value_lists)):
            params = dict(zip(keys, combo, strict=True))
            trials.append(TrialConfig(trial_id=trial_id, params=params))

        self._trials: tuple[TrialConfig, ...] = tuple(trials)

    @property
    def trials(self) -> tuple[TrialConfig, ...]:
        """All expanded `TrialConfig` instances in deterministic order."""
        return self._trials

    @property
    def total_trials(self) -> int:
        """Total number of expanded hyperparameter trials."""
        return len(self._trials)

    def plan_waves(self, world_size: int | VirtualGrid) -> list[WaveSchedule]:
        """Partition all trials into sequential waves of at most `world_size` ranks.

        Args:
            world_size: Number of virtual ranks available, or a `VirtualGrid` instance.

        Returns:
            Ordered list of `WaveSchedule` instances.
        """
        resolved_world_size = (
            world_size.world_size if isinstance(world_size, VirtualGrid) else world_size
        )
        if resolved_world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {resolved_world_size}.")

        waves: list[WaveSchedule] = []
        total = len(self._trials)
        for wave_index, start in enumerate(range(0, total, resolved_world_size)):
            chunk = self._trials[start : start + resolved_world_size]
            active_count = len(chunk)
            rank_to_trial = {rank: trial for rank, trial in enumerate(chunk)}
            active_ranks = tuple(range(active_count))
            idle_ranks = tuple(range(active_count, resolved_world_size))
            waves.append(
                WaveSchedule(
                    wave_index=wave_index,
                    rank_to_trial=rank_to_trial,
                    active_ranks=active_ranks,
                    idle_ranks=idle_ranks,
                )
            )
        return waves
