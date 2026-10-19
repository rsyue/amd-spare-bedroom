"""Hyperparameter grid expansion and multi-rank sequential wave scheduling.

Provides deterministic Cartesian search space expansion (`HyperparameterGrid`,
`TrialConfig`) and wave partitioning (`WaveSchedule`) across `VirtualGrid` ranks.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
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
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.params, MappingProxyType):
            object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True)
class WaveSchedule:
    """Assignment of hyperparameter trials to virtual ranks for one execution wave.

    Attributes:
        wave_index: Zero-indexed wave position (`0 .. num_waves - 1`).
        rank_to_trial: Mapping from active virtual rank index (`0 .. K - 1`) to
            its assigned `TrialConfig`, where K = len(active_ranks).
        active_ranks: Tuple of virtual rank indices executing a trial in this wave.
        idle_ranks: Tuple of virtual rank indices with no assigned trial in this wave.
    """

    wave_index: int
    rank_to_trial: Mapping[int, TrialConfig]
    active_ranks: tuple[int, ...]
    idle_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rank_to_trial, MappingProxyType):
            object.__setattr__(self, "rank_to_trial", MappingProxyType(dict(self.rank_to_trial)))


class HyperparameterGrid:
    """Deterministic Cartesian hyperparameter search space and wave scheduler.

    Expands a dictionary of hyperparameter candidate sequences into `TrialConfig`
    objects and partitions them into sequential waves of up to `world_size`
    concurrent virtual ranks.
    """

    def __init__(self, param_grid: Mapping[str, Sequence[Any]]) -> None:
        """Initialize and eagerly expand the Cartesian hyperparameter search space.

        Args:
            param_grid: Mapping of hyperparameter names to candidate value sequences.

        Raises:
            TypeError: If param_grid is not a Mapping, if any key is not a string,
                or if candidate values are strings/bytes or not sequences/iterables.
            ValueError: If param_grid is empty, or any candidate sequence is empty.
        """
        if not isinstance(param_grid, Mapping):
            raise TypeError(f"param_grid must be a Mapping, got {type(param_grid).__name__}.")
        if not param_grid:
            raise ValueError("param_grid must not be empty.")

        keys: list[str] = []
        value_lists: list[list[Any]] = []
        for key, candidates in param_grid.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Hyperparameter key must be a str, got {type(key).__name__} ({key!r})."
                )
            if not key.strip():
                raise ValueError("Hyperparameter key must not be empty or whitespace.")
            if isinstance(candidates, (str, bytes)):
                raise TypeError(
                    f"Candidate values for {key!r} must be a non-string Sequence, "
                    f"got {type(candidates).__name__}."
                )
            try:
                vals = [copy.deepcopy(c) for c in candidates]
            except TypeError:
                raise TypeError(
                    f"Candidate values for {key!r} must be an iterable sequence, "
                    f"got {type(candidates).__name__}."
                ) from None

            if len(vals) == 0:
                raise ValueError(
                    f"Hyperparameter {key!r} must contain at least one candidate value."
                )
            keys.append(key)
            value_lists.append(vals)

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

        Raises:
            TypeError: If world_size is not an int or VirtualGrid, or is a bool.
            ValueError: If world_size < 1.
        """
        resolved_world_size = (
            world_size.world_size if isinstance(world_size, VirtualGrid) else world_size
        )
        if isinstance(resolved_world_size, bool) or not isinstance(resolved_world_size, int):
            raise TypeError(
                f"world_size must be an int or VirtualGrid, got {type(world_size).__name__}."
            )
        if resolved_world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {resolved_world_size}.")

        waves: list[WaveSchedule] = []
        total_trials = len(self._trials)
        for wave_index, start in enumerate(range(0, total_trials, resolved_world_size)):
            wave_trials = self._trials[start : start + resolved_world_size]
            active_count = len(wave_trials)
            rank_to_trial = {rank: trial for rank, trial in enumerate(wave_trials)}
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
