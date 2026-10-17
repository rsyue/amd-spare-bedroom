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
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from amd_spare_bedroom.amp import AmpGradScalerState, resolve_amp_dtype
from amd_spare_bedroom.checkpoint import METRIC_DIRECTIONS, BestCheckpointSaver
from amd_spare_bedroom.grid import VirtualGrid, seed_everything
from amd_spare_bedroom.metrics import AverageMode, compute_classification_metrics


@dataclass(frozen=True)
class TrialConfig:
    """A single hyperparameter trial configuration in a grid search."""

    trial_id: int
    params: dict[str, Any]


@dataclass(frozen=True)
class WaveSchedule:
    """Assignment of hyperparameter trials to virtual ranks for one execution wave."""

    wave_index: int
    rank_to_trial: dict[int, TrialConfig]
    active_ranks: tuple[int, ...]
    idle_ranks: tuple[int, ...]


@dataclass(frozen=True)
class TrialResult:
    """Recorded metrics and best-epoch summary for a completed hyperparameter trial."""

    trial: TrialConfig
    wave_index: int
    assigned_rank: int
    epoch_metrics: tuple[dict[str, float], ...]
    best_epoch: int
    best_score: float
    best_metrics: dict[str, float]


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
        """Partition all trials into sequential waves of at most `world_size` ranks."""
        resolved_world_size = (
            world_size.world_size if isinstance(world_size, VirtualGrid) else world_size
        )
        if resolved_world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {resolved_world_size}.")

        waves: list[WaveSchedule] = []
        for wave_index, start in enumerate(range(0, len(self._trials), resolved_world_size)):
            chunk = self._trials[start : start + resolved_world_size]
            active_count = len(chunk)
            waves.append(
                WaveSchedule(
                    wave_index=wave_index,
                    rank_to_trial=dict(enumerate(chunk)),
                    active_ranks=tuple(range(active_count)),
                    idle_ranks=tuple(range(active_count, resolved_world_size)),
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
    def capture(cls) -> _RankRNGState:
        return cls(
            py_state=random.getstate(),
            np_state=np.random.get_state(),
            cpu_state=torch.get_rng_state(),
            cuda_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
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
        self.grid = grid
        self.hyperparameter_grid = (
            param_grid
            if isinstance(param_grid, HyperparameterGrid)
            else HyperparameterGrid(param_grid)
        )
        self.metric_name = metric_name
        if higher_is_better is not None:
            self.higher_is_better = higher_is_better
        elif checkpoint_saver is not None and checkpoint_saver.metric_name == metric_name:
            self.higher_is_better = checkpoint_saver.mode == "max"
        else:
            self.higher_is_better = METRIC_DIRECTIONS.get(metric_name, "max") == "max"

        self.amp_dtype = resolve_amp_dtype(amp_dtype)
        self.max_grad_norm = max_grad_norm
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
            labels = cast(torch.Tensor, device_batch.get("labels", device_batch.get("label")))
            outputs: Any = model(**device_batch)
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            inputs = batch[0].to(device) if isinstance(batch[0], torch.Tensor) else batch[0]
            labels = batch[1].to(device) if isinstance(batch[1], torch.Tensor) else batch[1]
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
                raw_map = raw_metrics.to_dict() if hasattr(raw_metrics, "to_dict") else raw_metrics
                metrics_dict = {str(k): float(v) for k, v in raw_map.items() if v is not None}
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
                    k: float(v) for k, v in cls_metrics.to_dict().items() if v is not None
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
                rng=_RankRNGState.capture(),
            )
        return states

    def _train_step_rank(
        self, state: _ActiveTrialState, batch: Any, target_device: torch.device
    ) -> None:
        state.optimizer.zero_grad(set_to_none=True)
        with state.scaler.autocast_context():
            loss, _, _ = self._forward_and_loss(state.replica, batch, target_device)
        state.scaler.backward(state.rank, loss)
        state.scaler.unscale_and_check_finite({state.rank: state.replica.parameters()})

        for param in state.replica.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        if self.max_grad_norm is not None and self.max_grad_norm > 0:
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
        """Execute all hyperparameter waves and return the aggregated `GridSearchResult`."""
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

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
                                state.rng = _RankRNGState.capture()
                            if exhausted:
                                break
                        if exhausted:
                            break

                        step += 1
                        for rank in wave.active_ranks:
                            state = states[rank]
                            state.rng.restore()
                            self._train_step_rank(state, batches[rank], target_device)
                            state.rng = _RankRNGState.capture()

                    for rank in wave.active_ranks:
                        state = states[rank]
                        metrics_dict = self._evaluate_replica(
                            state=state,
                            eval_dataset=val_dataset,
                            batch_size=batch_size,
                            device=target_device,
                            average=average,
                            eval_fn=eval_fn,
                        )
                        if self.metric_name not in metrics_dict:
                            raise KeyError(
                                f"Monitored metric {self.metric_name!r} not found in evaluation "
                                f"metrics {sorted(metrics_dict.keys())}."
                            )
                        score = float(metrics_dict[self.metric_name])
                        state.epoch_history.append(metrics_dict)

                        if self._is_better(score, state.best_score):
                            state.best_score = score
                            state.best_epoch = epoch + 1
                            state.best_metrics = dict(metrics_dict)

                        if self.checkpoint_saver is not None:
                            self.checkpoint_saver.save_if_best(
                                model=state.replica,
                                metrics=metrics_dict,
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
                states.clear()
                gc.collect()
                if self.grid.device_type == "cuda" and torch.cuda.is_available():
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
