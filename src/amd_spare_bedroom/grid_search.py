"""Hyperparameter grid expansion, sequential wave scheduling, and isolated trial execution.

Provides deterministic Cartesian search space expansion (`HyperparameterGrid`,
`TrialConfig`), wave partitioning (`WaveSchedule`), and the multi-rank execution
engine (`VirtualGridSearch`, `TrialResult`, `GridSearchResult`).
"""

from __future__ import annotations

import copy
import gc
import itertools
import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from amd_spare_bedroom.amp import (
    ROCM_GRAD_FINITE_BOUND,
    AmpGradScalerState,
    resolve_amp_dtype,
)
from amd_spare_bedroom.checkpoint import METRIC_DIRECTIONS, BestCheckpointSaver
from amd_spare_bedroom.grid import VirtualGrid, seed_everything
from amd_spare_bedroom.metrics import AverageMode, compute_classification_metrics


def parse_float_grid(
    raw: str,
    *,
    name: str = "grid",
    min_val: float | None = None,
    allow_zero: bool = True,
    allow_negative: bool = False,
    deduplicate: bool = True,
) -> list[float]:
    """Parse a comma- or whitespace-separated string into a validated list of floats.

    Args:
        raw: Comma- and/or whitespace-delimited string of float values.
        name: Name of the parameter or CLI flag for error messaging.
        min_val: Optional minimum value threshold.
        allow_zero: Whether zero is allowed when min_val is 0.0.
        allow_negative: Whether negative values are allowed.
        deduplicate: Whether to remove duplicate values while preserving order.

    Returns:
        List of parsed and validated floating point numbers.

    Raises:
        TypeError: If raw is not a string.
        ValueError: If empty, non-numeric tokens exist, non-finite values (NaN/Inf) exist,
            or values violate bounds.
    """
    if not isinstance(raw, str):
        raise TypeError(f"{name} must be a string, got {type(raw).__name__}.")
    tokens = [tok for part in raw.split(",") for tok in part.split() if tok.strip()]
    if not tokens:
        raise ValueError(f"{name} must contain at least one numeric value.")

    values: list[float] = []
    seen: set[float] = set()
    for tok in tokens:
        try:
            val = float(tok)
        except ValueError:
            raise ValueError(f"Invalid numeric value {tok!r} in {name}.") from None

        if not math.isfinite(val):
            raise ValueError(f"{name} values must be finite numbers, got {tok!r}.")

        if not allow_negative and val < 0.0:
            raise ValueError(f"{name} values must be non-negative, got {val}.")

        if min_val is not None:
            if not allow_zero and val <= min_val:
                raise ValueError(
                    f"Values in {name} must be strictly greater than {min_val}, got {val}."
                )
            if allow_zero and val < min_val:
                raise ValueError(f"Values in {name} must be >= {min_val}, got {val}.")

        if deduplicate:
            if val in seen:
                continue
            seen.add(val)
        values.append(val)

    return values


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


@dataclass(frozen=True)
class TrialResult:
    """Recorded metrics and best-epoch summary for a completed hyperparameter trial."""

    trial: TrialConfig
    wave_index: int
    assigned_rank: int
    epoch_metrics: tuple[Mapping[str, float], ...]
    best_epoch: int
    best_score: float
    best_metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.best_metrics, MappingProxyType):
            object.__setattr__(self, "best_metrics", MappingProxyType(dict(self.best_metrics)))
        frozen_epoch_metrics = tuple(
            m if isinstance(m, MappingProxyType) else MappingProxyType(dict(m))
            for m in self.epoch_metrics
        )
        object.__setattr__(self, "epoch_metrics", frozen_epoch_metrics)


@dataclass(frozen=True)
class GridSearchResult:
    """Aggregated leaderboard across all waves of a `VirtualGridSearch` run."""

    trials: tuple[TrialResult, ...]
    best_trial: TrialResult
    metric_name: str
    higher_is_better: bool
    num_waves: int

    def format_winner_announcement(self) -> str:
        """Format the canonical winning combination announcement string."""
        return BestCheckpointSaver.format_winner(
            hyperparameters=self.best_trial.trial.params,
            metric_name=self.metric_name,
            best_score=self.best_trial.best_score,
        )


class HyperparameterGrid:
    """Deterministic Cartesian hyperparameter search space and wave scheduler."""

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

        self._trials: tuple[TrialConfig, ...] = tuple(
            TrialConfig(trial_id=i, params=dict(zip(keys, combo, strict=True)))
            for i, combo in enumerate(itertools.product(*value_lists))
        )

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


@dataclass
class _RankRNGState:
    """Saved Python, NumPy, and PyTorch RNG state for a single active rank trial."""

    py_state: tuple[Any, ...]
    np_state: Any
    cpu_state: torch.Tensor
    cuda_states: list[torch.Tensor] | None

    @classmethod
    def capture(cls, device_type: str = "cpu") -> _RankRNGState:
        capture_cuda = (
            torch.device(device_type).type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
        )
        return cls(
            py_state=random.getstate(),
            np_state=np.random.get_state(),
            cpu_state=torch.get_rng_state(),
            cuda_states=torch.cuda.get_rng_state_all() if capture_cuda else None,
        )

    def restore(self) -> None:
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.cpu_state)
        if self.cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_states)


@dataclass
class _ActiveTrialState:
    """Per-rank mutable state during a single wave of `VirtualGridSearch`."""

    rank: int
    trial: TrialConfig
    replica: nn.Module
    optimizer: torch.optim.Optimizer
    scaler: AmpGradScalerState
    rng: _RankRNGState
    epoch_history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 1
    best_score: float | None = None
    best_metrics: dict[str, float] = field(default_factory=dict)


EvalCallback = Callable[
    [nn.Module, Dataset[Any], int, torch.device, torch.dtype | None],
    Mapping[str, float] | Any,
]


class VirtualGridSearch:
    """Multi-rank hyperparameter grid search execution engine."""

    def __init__(
        self,
        base_model: nn.Module,
        grid: VirtualGrid,
        param_grid: Mapping[str, Sequence[Any]] | HyperparameterGrid,
        *,
        metric_name: str = "f1",
        higher_is_better: bool | None = None,
        amp_dtype: torch.dtype | str | None = None,
        max_grad_norm: float | None = 1.0,
        seed: int = 42,
        checkpoint_saver: BestCheckpointSaver | None = None,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
    ) -> None:
        """Initialize the multi-rank hyperparameter grid search execution engine.

        Args:
            base_model: Base neural network module whose initial weights are replicated.
            grid: VirtualGrid managing emulated ranks and target device.
            param_grid: Hyperparameter candidate mapping or expanded HyperparameterGrid.
            metric_name: Validation metric name to optimize (e.g. 'f1', 'loss', 'accuracy').
            higher_is_better: Whether higher metric values represent improvement.
            amp_dtype: Optional mixed precision dtype ('bfloat16', 'float16', or None).
            max_grad_norm: Maximum gradient norm for clipping, or None to disable.
            seed: Master random seed for deterministic replica and data generator initialization.
            checkpoint_saver: Optional BestCheckpointSaver to atomically record winning checkpoints.
            optimizer_cls: Optimizer class instantiated per trial (defaults to torch.optim.AdamW).

        Raises:
            TypeError: If base_model, grid, or optimizer_cls are of invalid types.
            ValueError: If metric_name is empty, seed is out of range, max_grad_norm <= 0,
                or checkpoint_saver monitored metric does not match metric_name.
        """
        if not isinstance(base_model, nn.Module):
            raise TypeError(f"base_model must be an nn.Module, got {type(base_model).__name__}.")
        if not isinstance(grid, VirtualGrid):
            raise TypeError(f"grid must be a VirtualGrid, got {type(grid).__name__}.")
        if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed < 2**32):
            raise ValueError(f"seed must be an int between 0 and 2**32 - 1, got {seed!r}.")
        if max_grad_norm is not None:
            if (
                isinstance(max_grad_norm, bool)
                or not isinstance(max_grad_norm, (int, float))
                or max_grad_norm <= 0
                or not math.isfinite(max_grad_norm)
            ):
                raise ValueError(
                    f"max_grad_norm must be a positive finite float or None, got {max_grad_norm!r}."
                )
        if not (
            isinstance(optimizer_cls, type) and issubclass(optimizer_cls, torch.optim.Optimizer)
        ):
            raise TypeError(
                f"optimizer_cls must be a subclass of torch.optim.Optimizer, got {optimizer_cls!r}."
            )
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError(f"metric_name must be a non-empty str, got {metric_name!r}.")

        self.grid = grid
        self.hyperparameter_grid = (
            param_grid
            if isinstance(param_grid, HyperparameterGrid)
            else HyperparameterGrid(param_grid)
        )
        self.metric_name = metric_name.strip().lower()

        if checkpoint_saver is not None and checkpoint_saver.metric_name != self.metric_name:
            raise ValueError(
                f"checkpoint_saver monitored metric {checkpoint_saver.metric_name!r} "
                f"does not match grid search metric {self.metric_name!r}."
            )
        if higher_is_better is not None:
            self.higher_is_better = higher_is_better
        elif checkpoint_saver is not None:
            self.higher_is_better = checkpoint_saver.mode == "max"
        else:
            self.higher_is_better = METRIC_DIRECTIONS.get(self.metric_name, "max") == "max"

        self.amp_dtype = resolve_amp_dtype(amp_dtype)
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
        self.seed = seed
        self.checkpoint_saver = checkpoint_saver
        self.optimizer_cls = optimizer_cls

        self._base_model_template: nn.Module = copy.deepcopy(base_model).cpu()
        self._initial_state_dict: dict[str, torch.Tensor] = {
            k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()
        }

    def _is_better(self, candidate: float, reference: float | None) -> bool:
        if not math.isfinite(candidate):
            return False
        if reference is None or not math.isfinite(reference):
            return True
        return candidate > reference if self.higher_is_better else candidate < reference

    @staticmethod
    def _forward_and_loss(
        model: nn.Module,
        batch: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass on a mapping or `(inputs, labels)` batch and return `(loss, logits_cpu, labels_cpu)`."""
        if isinstance(batch, Mapping):
            device_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            labels = device_batch.get("labels", device_batch.get("label"))
            if labels is None or not isinstance(labels, torch.Tensor):
                raise KeyError("Batch mapping must contain a Tensor under 'labels' or 'label' key.")
            outputs: Any = model(**device_batch)
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            inputs = batch[0].to(device) if isinstance(batch[0], torch.Tensor) else batch[0]
            labels = batch[1].to(device) if isinstance(batch[1], torch.Tensor) else batch[1]
            if not isinstance(labels, torch.Tensor):
                raise TypeError(f"Batch labels must be a Tensor, got {type(labels).__name__}.")
            outputs = model(inputs)
        else:
            raise TypeError(f"Unsupported batch type for VirtualGridSearch: {type(batch)!r}")

        logits = cast(
            torch.Tensor,
            outputs.logits
            if hasattr(outputs, "logits")
            else (outputs["logits"] if isinstance(outputs, Mapping) else outputs),
        )
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = cast(torch.Tensor, outputs.loss)
        elif isinstance(outputs, Mapping) and outputs.get("loss") is not None:
            loss = cast(torch.Tensor, outputs["loss"])
        else:
            loss = F.cross_entropy(logits, labels)

        if loss.ndim > 0:
            loss = loss.mean()

        return loss, logits.detach().cpu(), labels.detach().cpu()

    def _evaluate_replica(
        self,
        state: _ActiveTrialState,
        eval_dataset: Dataset[Any],
        batch_size: int,
        device: torch.device,
        average: AverageMode,
        eval_fn: EvalCallback | None,
    ) -> dict[str, float]:
        """Evaluate a single rank's model replica in `.eval()` mode."""
        model = state.replica
        was_training = model.training
        model.eval()
        try:
            if eval_fn is not None:
                raw_metrics = eval_fn(model, eval_dataset, batch_size, device, self.amp_dtype)
                if hasattr(raw_metrics, "to_dict"):
                    raw_map = raw_metrics.to_dict()
                elif isinstance(raw_metrics, Mapping):
                    raw_map = raw_metrics
                else:
                    raise TypeError(
                        f"eval_fn must return a Mapping or an object with to_dict(), "
                        f"got {type(raw_metrics).__name__}."
                    )
                metrics_dict = {
                    str(k).lower(): float(v) for k, v in raw_map.items() if v is not None
                }
            else:
                eval_loader: DataLoader[Any] = DataLoader(
                    eval_dataset, batch_size=batch_size, shuffle=False
                )
                all_logits: list[torch.Tensor] = []
                all_labels: list[torch.Tensor] = []
                total_loss = 0.0
                total_examples = 0

                with torch.no_grad():
                    for batch in eval_loader:
                        with state.scaler.autocast_context():
                            loss, logits_cpu, labels_cpu = self._forward_and_loss(
                                model, batch, device
                            )
                        count = int(labels_cpu.shape[0]) if labels_cpu.ndim > 0 else 1
                        total_loss += float(loss.detach().cpu().item()) * count
                        total_examples += count
                        all_logits.append(logits_cpu)
                        all_labels.append(labels_cpu.reshape(-1))

                if not all_logits:
                    raise ValueError("Evaluation dataset produced no batches.")

                avg_loss = total_loss / max(total_examples, 1)
                cls_metrics = replace(
                    compute_classification_metrics(
                        preds=torch.cat(all_logits, dim=0),
                        labels=torch.cat(all_labels, dim=0),
                        average=average,
                    ),
                    loss=avg_loss,
                )
                metrics_dict = {
                    k.lower(): float(v) for k, v in cls_metrics.to_dict().items() if v is not None
                }

            if "loss" in metrics_dict and "eval_loss" not in metrics_dict:
                metrics_dict["eval_loss"] = metrics_dict["loss"]
            elif "eval_loss" in metrics_dict and "loss" not in metrics_dict:
                metrics_dict["loss"] = metrics_dict["eval_loss"]
            return metrics_dict
        finally:
            model.train(was_training)

    def _init_wave_states(
        self, wave: WaveSchedule, target_device: torch.device
    ) -> dict[int, _ActiveTrialState]:
        states: dict[int, _ActiveTrialState] = {}
        for rank in wave.active_ranks:
            trial = wave.rank_to_trial[rank]
            seed_everything(self.seed)
            replica = copy.deepcopy(self._base_model_template).to(target_device)
            replica.load_state_dict(self._initial_state_dict)
            replica.train()
            optimizer = self.optimizer_cls(replica.parameters(), **trial.params)
            scaler = AmpGradScalerState(
                device_type=self.grid.device_type,
                ranks=[rank],
                amp_dtype=self.amp_dtype,
            )
            states[rank] = _ActiveTrialState(
                rank=rank,
                trial=trial,
                replica=replica,
                optimizer=optimizer,
                scaler=scaler,
                rng=_RankRNGState.capture(self.grid.device_type),
            )
        return states

    def _train_step_rank(
        self, state: _ActiveTrialState, batch: Any, target_device: torch.device
    ) -> None:
        state.optimizer.zero_grad(set_to_none=True)
        with state.scaler.autocast_context():
            loss, _, _ = self._forward_and_loss(state.replica, batch, target_device)
        state.scaler.backward(state.rank, loss)
        is_finite = state.scaler.unscale_and_check_finite({state.rank: state.replica.parameters()})

        for param in state.replica.parameters():
            if param.grad is not None:
                param.grad.nan_to_num_(
                    nan=0.0,
                    posinf=ROCM_GRAD_FINITE_BOUND,
                    neginf=-ROCM_GRAD_FINITE_BOUND,
                )

        if is_finite and self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(state.replica.parameters(), self.max_grad_norm)

        state.scaler.step_optimizers(
            {state.rank: state.optimizer},
            {state.rank: state.replica.parameters()},
        )

    def run(
        self,
        train_dataset: Dataset[Any],
        eval_dataset: Dataset[Any] | None = None,
        *,
        batch_size: int = 8,
        epochs: int = 1,
        max_steps: int | None = None,
        eval_fn: EvalCallback | None = None,
        average: AverageMode = "auto",
    ) -> GridSearchResult:
        """Execute all hyperparameter waves and return the aggregated `GridSearchResult`.

        Args:
            train_dataset: Dataset used for per-rank training across epochs.
            eval_dataset: Dataset used for validation (defaults to train_dataset if None).
            batch_size: Per-rank batch size for training and evaluation.
            epochs: Number of training epochs (must be >= 1).
            max_steps: Maximum training steps per epoch, or None for full epochs.
            eval_fn: Optional custom evaluation callback taking (model, dataset, ...).
            average: Averaging mode for classification metrics ('micro', 'macro', 'weighted', 'auto').

        Returns:
            GridSearchResult containing all completed TrialResult objects, the overall
            best TrialResult, and leaderboard metadata.

        Raises:
            ValueError: If epochs < 1, batch_size < 1, max_steps < 1, or datasets are empty.
            KeyError: If monitored metric_name is missing from evaluation metrics.
        """
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"epochs must be an int >= 1, got {epochs!r}.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be an int >= 1, got {batch_size!r}.")
        if max_steps is not None:
            if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
                raise ValueError(f"max_steps must be an int >= 1 or None, got {max_steps!r}.")

        try:
            if len(train_dataset) == 0:  # type: ignore[arg-type]
                raise ValueError("train_dataset must not be empty.")
            if eval_dataset is not None and len(eval_dataset) == 0:  # type: ignore[arg-type]
                raise ValueError("eval_dataset must not be empty.")
        except TypeError:
            pass

        val_dataset = eval_dataset if eval_dataset is not None else train_dataset
        target_device = torch.device(self.grid.device_type)
        waves = self.hyperparameter_grid.plan_waves(self.grid.world_size)
        has_max_steps = max_steps is not None and max_steps > 0
        all_trial_results: list[TrialResult] = []

        for wave in waves:
            states = self._init_wave_states(wave, target_device)
            try:
                for epoch in range(epochs):
                    rank_iters: dict[int, Iterator[Any]] = {}
                    for rank in wave.active_ranks:
                        gen = torch.Generator()
                        gen.manual_seed(self.seed + epoch)
                        loader: DataLoader[Any] = DataLoader(
                            train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            generator=gen,
                        )
                        rank_iters[rank] = iter(loader)

                    step = 0
                    while not has_max_steps or (max_steps is not None and step < max_steps):
                        batches: dict[int, Any] = {}
                        exhausted = False
                        for rank in wave.active_ranks:
                            state = states[rank]
                            state.rng.restore()
                            try:
                                batches[rank] = next(rank_iters[rank])
                            except StopIteration:
                                exhausted = True
                            finally:
                                state.rng = _RankRNGState.capture(self.grid.device_type)
                            if exhausted:
                                break
                        if exhausted:
                            break

                        step += 1
                        for rank in wave.active_ranks:
                            state = states[rank]
                            state.rng.restore()
                            self._train_step_rank(state, batches[rank], target_device)
                            state.rng = _RankRNGState.capture(self.grid.device_type)

                    for rank in wave.active_ranks:
                        state = states[rank]
                        state.rng.restore()
                        try:
                            metrics_dict = self._evaluate_replica(
                                state=state,
                                eval_dataset=val_dataset,
                                batch_size=batch_size,
                                device=target_device,
                                average=average,
                                eval_fn=eval_fn,
                            )
                        finally:
                            state.rng = _RankRNGState.capture(self.grid.device_type)

                        norm_metrics = {k.lower(): float(v) for k, v in metrics_dict.items()}
                        if self.metric_name not in norm_metrics:
                            raise KeyError(
                                f"Monitored metric {self.metric_name!r} not found in evaluation "
                                f"metrics {sorted(norm_metrics.keys())}."
                            )
                        score = float(norm_metrics[self.metric_name])
                        state.epoch_history.append(norm_metrics)

                        if self._is_better(score, state.best_score):
                            state.best_score = score
                            state.best_epoch = epoch + 1
                            state.best_metrics = dict(norm_metrics)

                        if self.checkpoint_saver is not None:
                            self.checkpoint_saver.save_if_best(
                                model=state.replica,
                                metrics=norm_metrics,
                                epoch=epoch + 1,
                                hyperparameters=dict(state.trial.params),
                            )

                for rank in wave.active_ranks:
                    state = states[rank]
                    all_trial_results.append(
                        TrialResult(
                            trial=state.trial,
                            wave_index=wave.wave_index,
                            assigned_rank=rank,
                            epoch_metrics=tuple(state.epoch_history),
                            best_epoch=state.best_epoch,
                            best_score=(
                                state.best_score
                                if state.best_score is not None
                                else float(state.epoch_history[-1][self.metric_name])
                            ),
                            best_metrics=(
                                state.best_metrics
                                if state.best_metrics
                                else dict(state.epoch_history[-1])
                            ),
                        )
                    )
            finally:
                for s in states.values():
                    s.optimizer.zero_grad(set_to_none=True)
                    s.optimizer.state.clear()
                    s.replica.to("cpu")
                states.clear()
                gc.collect()
                if torch.device(self.grid.device_type).type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        best_trial = all_trial_results[0]
        for candidate_result in all_trial_results[1:]:
            if self._is_better(candidate_result.best_score, best_trial.best_score):
                best_trial = candidate_result

        return GridSearchResult(
            trials=tuple(all_trial_results),
            best_trial=best_trial,
            metric_name=self.metric_name,
            higher_is_better=self.higher_is_better,
            num_waves=len(waves),
        )
