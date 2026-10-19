"""Tests for hyperparameter grid expansion and sequential wave scheduling."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset

from amd_spare_bedroom.checkpoint import BestCheckpointSaver
from amd_spare_bedroom.grid import VirtualGrid, seed_everything
from amd_spare_bedroom.grid_search import (
    GridSearchResult,
    HyperparameterGrid,
    TrialConfig,
    TrialResult,
    VirtualGridSearch,
    WaveSchedule,
)


def _make_5x5_grid() -> HyperparameterGrid:
    return HyperparameterGrid(
        {
            "lr": [1e-5, 2e-5, 3e-5, 5e-5, 1e-4],
            "weight_decay": [0.0, 1e-4, 1e-3, 1e-2, 1e-1],
        }
    )


def _make_synthetic_classification_data(
    seed: int = 123,
) -> tuple[nn.Module, TensorDataset, TensorDataset]:
    seed_everything(seed)
    base_model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    x_train = torch.randn(24, 8)
    y_train = (x_train[:, 0] + 0.5 * x_train[:, 1] > 0).long()
    x_val = torch.randn(16, 8)
    y_val = (x_val[:, 0] + 0.5 * x_val[:, 1] > 0).long()
    return base_model, TensorDataset(x_train, y_train), TensorDataset(x_val, y_val)


class _DictLinearDataset(Dataset[dict[str, torch.Tensor]]):
    """Mapping-batch dataset for testing dict-style forward/eval in VirtualGridSearch."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self.features[idx], "labels": self.labels[idx]}


class _DictLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(8, 2)

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        logits = self.net(x)
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(logits, labels)
        return out

    def save_pretrained(self, save_directory: str | Path, safe_serialization: bool = True) -> None:
        assert safe_serialization
        target = Path(save_directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text('{"model_type": "dict_linear"}\n', encoding="utf-8")
        (target / "model.safetensors").write_bytes(b"safetensors")


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


def test_trial_config_and_wave_schedule_immutability() -> None:
    """Verify TrialConfig.params and WaveSchedule.rank_to_trial are read-only mapping proxies."""
    grid = HyperparameterGrid({"lr": [1e-4]})
    trial = grid.trials[0]

    with pytest.raises(TypeError):
        trial.params["lr"] = 1.0  # type: ignore[index]

    wave = grid.plan_waves(world_size=2)[0]
    with pytest.raises(TypeError):
        wave.rank_to_trial[0] = trial  # type: ignore[index]


def test_hyperparameter_grid_heterogeneous_and_generator_candidates() -> None:
    """Verify support for heterogeneous parameter types and generator candidate iterables."""
    grid = HyperparameterGrid(
        {
            "model_type": ["bert", "roberta"],
            "use_bias": [True, False],
            "hidden_dims": [(64, 64)],
            "lr": (val for val in [1e-5, 2e-5]),  # type: ignore[dict-item]
        }
    )
    assert grid.total_trials == 8
    first = grid.trials[0]
    assert first.params["model_type"] == "bert"
    assert first.params["use_bias"] is True
    assert first.params["hidden_dims"] == (64, 64)
    assert first.params["lr"] == 1e-5


def test_hyperparameter_grid_extended_validation() -> None:
    """Verify defensive type and boundary checks for param_grid, keys, and world_size."""
    with pytest.raises(TypeError, match="param_grid must be a Mapping"):
        HyperparameterGrid([("lr", [1e-5])])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Hyperparameter key must be a str"):
        HyperparameterGrid({100: [1e-5]})  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="must not be empty or whitespace"):
        HyperparameterGrid({"   ": [1e-5]})

    with pytest.raises(TypeError, match="must be a non-string Sequence"):
        HyperparameterGrid({"model": "bert"})

    grid = HyperparameterGrid({"lr": [1e-5, 2e-5]})
    with pytest.raises(TypeError, match="world_size must be an int or VirtualGrid"):
        grid.plan_waves(True)

    with pytest.raises(TypeError, match="world_size must be an int or VirtualGrid"):
        grid.plan_waves(2.5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="world_size must be an int or VirtualGrid"):
        grid.plan_waves("4")  # type: ignore[arg-type]


def test_plan_waves_single_rank_and_single_trial() -> None:
    """Verify plan_waves behavior when world_size=1 or total_trials=1."""
    single_trial_grid = HyperparameterGrid({"lr": [1e-4]})
    waves_4ranks = single_trial_grid.plan_waves(world_size=4)
    assert len(waves_4ranks) == 1
    assert waves_4ranks[0].active_ranks == (0,)
    assert waves_4ranks[0].idle_ranks == (1, 2, 3)

    three_trials_grid = HyperparameterGrid({"lr": [1e-4, 2e-4, 3e-4]})
    waves_1rank = three_trials_grid.plan_waves(world_size=1)
    assert len(waves_1rank) == 3
    for idx, wave in enumerate(waves_1rank):
        assert wave.wave_index == idx
        assert wave.active_ranks == (0,)
        assert wave.idle_ranks == ()
        assert wave.rank_to_trial[0].trial_id == idx


def test_virtual_grid_search_world_size_invariance() -> None:
    """Seam 1: 4 trials on world_size=4 (1 wave) vs world_size=2 (2 waves) match exactly."""
    base_model, train_ds, val_ds = _make_synthetic_classification_data(seed=99)
    param_grid = {
        "lr": [1e-3, 5e-2],
        "weight_decay": [0.0, 1e-2],
    }

    grid_4 = VirtualGrid(world_size=4, total_memory_bytes=4 * 1024**3, device_type="cpu", seed=77)
    search_1wave = VirtualGridSearch(
        base_model=base_model,
        grid=grid_4,
        param_grid=param_grid,
        metric_name="f1",
        seed=77,
    )
    res_1wave = search_1wave.run(train_ds, val_ds, batch_size=8, epochs=2)

    grid_2 = VirtualGrid(world_size=2, total_memory_bytes=2 * 1024**3, device_type="cpu", seed=77)
    search_2waves = VirtualGridSearch(
        base_model=base_model,
        grid=grid_2,
        param_grid=param_grid,
        metric_name="f1",
        seed=77,
    )
    res_2waves = search_2waves.run(train_ds, val_ds, batch_size=8, epochs=2)

    assert isinstance(res_1wave, GridSearchResult)
    assert res_1wave.num_waves == 1
    assert res_2waves.num_waves == 2
    assert len(res_1wave.trials) == 4
    assert len(res_2waves.trials) == 4

    for t1, t2 in zip(res_1wave.trials, res_2waves.trials, strict=True):
        assert isinstance(t1, TrialResult)
        assert t1.trial == t2.trial
        assert t1.best_epoch == t2.best_epoch
        assert t1.best_score == pytest.approx(t2.best_score, abs=1e-6)
        assert len(t1.epoch_metrics) == len(t2.epoch_metrics) == 2
        for m1, m2 in zip(t1.epoch_metrics, t2.epoch_metrics, strict=True):
            for k in m1:
                assert m1[k] == pytest.approx(m2[k], abs=1e-6)

    assert res_1wave.best_trial.trial == res_2waves.best_trial.trial


def test_virtual_grid_search_partial_final_wave() -> None:
    """Seam 2: 5 trials on a 3-rank VirtualGrid runs 2 waves (3 active, then 2 active)."""
    base_model, train_ds, val_ds = _make_synthetic_classification_data(seed=101)
    param_grid = {
        "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "weight_decay": [0.0],
    }
    grid_3 = VirtualGrid(world_size=3, total_memory_bytes=3 * 1024**3, device_type="cpu", seed=42)
    search = VirtualGridSearch(
        base_model=base_model,
        grid=grid_3,
        param_grid=param_grid,
        metric_name="accuracy",
        seed=42,
    )
    result = search.run(train_ds, val_ds, batch_size=8, epochs=1)

    assert result.num_waves == 2
    assert len(result.trials) == 5
    assert [t.wave_index for t in result.trials] == [0, 0, 0, 1, 1]
    assert [t.assigned_rank for t in result.trials] == [0, 1, 2, 0, 1]


def test_virtual_grid_search_distinct_hyperparameter_trajectories(tmp_path: Path) -> None:
    """Seam 3: lr=0.0 leaves weights unchanged while lr=0.1 updates weights independently."""
    base_model, train_ds, val_ds = _make_synthetic_classification_data(seed=202)
    initial_params = {k: v.clone() for k, v in base_model.state_dict().items()}

    grid_2 = VirtualGrid(world_size=2, total_memory_bytes=2 * 1024**3, device_type="cpu", seed=55)
    observed_eval_weights: list[dict[str, torch.Tensor]] = []

    def _custom_eval(
        model: nn.Module,
        eval_dataset: Dataset[object],
        batch_size: int,
        device: torch.device,
        amp_dtype: torch.dtype | None,
    ) -> dict[str, float]:
        assert not model.training
        assert len(eval_dataset) == 16  # type: ignore[arg-type]
        assert batch_size == 8
        assert device == torch.device("cpu")
        assert amp_dtype is None
        snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
        observed_eval_weights.append(snapshot)
        max_delta = max(
            float((snapshot[k] - initial_params[k]).abs().max().item()) for k in initial_params
        )
        return {"f1": max_delta, "loss": 1.0 / (1.0 + max_delta)}

    search = VirtualGridSearch(
        base_model=base_model,
        grid=grid_2,
        param_grid={"lr": [0.0, 0.1], "weight_decay": [0.0]},
        metric_name="f1",
        seed=55,
    )
    result = search.run(
        train_ds,
        val_ds,
        batch_size=8,
        epochs=1,
        max_steps=2,
        eval_fn=_custom_eval,
    )

    assert len(observed_eval_weights) == 2
    # Rank 0 (lr=0.0) weights must equal initial weights exactly
    for k, init_tensor in initial_params.items():
        assert torch.allclose(observed_eval_weights[0][k], init_tensor, atol=1e-7)

    # Rank 1 (lr=0.1) weights must have diverged from initial weights
    assert any(
        not torch.allclose(observed_eval_weights[1][k], init_tensor, atol=1e-5)
        for k, init_tensor in initial_params.items()
    )
    assert result.trials[0].best_score == pytest.approx(0.0, abs=1e-7)
    assert result.trials[1].best_score > 0.0
    assert result.best_trial == result.trials[1]

    # Also verify mapping-batch dataset and BestCheckpointSaver integration
    seed_everything(303)
    dict_model = _DictLinearModel()
    x = torch.randn(16, 8)
    y = (x[:, 0] > 0).long()
    dict_ds = _DictLinearDataset(x, y)
    saver = BestCheckpointSaver(tmp_path / "ckpt", metric_name="loss", mode="min")
    dict_search = VirtualGridSearch(
        base_model=dict_model,
        grid=grid_2,
        param_grid=HyperparameterGrid({"lr": [1e-4, 5e-2]}),
        metric_name="loss",
        checkpoint_saver=saver,
        seed=55,
    )
    dict_res = dict_search.run(dict_ds, dict_ds, batch_size=8, epochs=1)
    assert not dict_res.higher_is_better
    assert saver.best_score == pytest.approx(dict_res.best_trial.best_score, abs=1e-6)


def test_trial_result_immutability() -> None:
    """Verify that TrialResult.best_metrics and epoch_metrics dictionaries cannot be mutated."""
    trial_cfg = TrialConfig(trial_id=0, params={"lr": 1e-4})
    result = TrialResult(
        trial=trial_cfg,
        wave_index=0,
        assigned_rank=0,
        epoch_metrics=({"f1": 0.8, "loss": 0.5},),
        best_epoch=1,
        best_score=0.8,
        best_metrics={"f1": 0.8, "loss": 0.5},
    )

    with pytest.raises(TypeError):
        result.best_metrics["f1"] = 0.9  # type: ignore[index]

    with pytest.raises(TypeError):
        result.epoch_metrics[0]["loss"] = 0.1  # type: ignore[index]


def test_virtual_grid_search_validations(tmp_path: Path) -> None:
    """Verify defensive type and boundary checks for VirtualGridSearch.__init__ and run()."""
    base_model, train_ds, val_ds = _make_synthetic_classification_data(seed=1)
    grid = VirtualGrid(world_size=2, total_memory_bytes=2 * 1024**3, device_type="cpu", seed=1)

    # __init__ validations
    with pytest.raises(TypeError, match="base_model must be an nn.Module"):
        VirtualGridSearch(base_model="not_a_model", grid=grid, param_grid={"lr": [1e-3]})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="grid must be a VirtualGrid"):
        VirtualGridSearch(base_model=base_model, grid="not_a_grid", param_grid={"lr": [1e-3]})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="seed must be an int"):
        VirtualGridSearch(base_model=base_model, grid=grid, param_grid={"lr": [1e-3]}, seed=-1)

    with pytest.raises(ValueError, match="max_grad_norm must be a positive finite float"):
        VirtualGridSearch(
            base_model=base_model, grid=grid, param_grid={"lr": [1e-3]}, max_grad_norm=-0.5
        )

    with pytest.raises(TypeError, match="optimizer_cls must be a subclass"):
        VirtualGridSearch(
            base_model=base_model,
            grid=grid,
            param_grid={"lr": [1e-3]},
            optimizer_cls=object,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="metric_name must be a non-empty str"):
        VirtualGridSearch(
            base_model=base_model, grid=grid, param_grid={"lr": [1e-3]}, metric_name="   "
        )

    saver = BestCheckpointSaver(tmp_path / "ckpt", metric_name="loss", mode="min")
    with pytest.raises(ValueError, match="does not match grid search metric"):
        VirtualGridSearch(
            base_model=base_model,
            grid=grid,
            param_grid={"lr": [1e-3]},
            metric_name="accuracy",
            checkpoint_saver=saver,
        )

    # run() validations
    search = VirtualGridSearch(base_model=base_model, grid=grid, param_grid={"lr": [1e-3]}, seed=1)
    with pytest.raises(ValueError, match="epochs must be an int >= 1"):
        search.run(train_ds, epochs=0)

    with pytest.raises(ValueError, match="batch_size must be an int >= 1"):
        search.run(train_ds, batch_size=0)

    with pytest.raises(ValueError, match="max_steps must be an int >= 1 or None"):
        search.run(train_ds, max_steps=0)

    empty_ds = TensorDataset(torch.empty(0, 8), torch.empty(0, dtype=torch.long))
    with pytest.raises(ValueError, match="train_dataset must not be empty"):
        search.run(empty_ds)

    with pytest.raises(ValueError, match="eval_dataset must not be empty"):
        search.run(train_ds, eval_dataset=empty_ds)


def test_virtual_grid_search_case_insensitivity_and_non_finite() -> None:
    """Verify case-insensitive metric normalization and non-finite score handling."""
    base_model, train_ds, val_ds = _make_synthetic_classification_data(seed=12)
    grid = VirtualGrid(world_size=2, total_memory_bytes=2 * 1024**3, device_type="cpu", seed=12)

    # Passing uppercase metric_name "F1" works cleanly
    search = VirtualGridSearch(
        base_model=base_model,
        grid=grid,
        param_grid={"lr": [1e-3, 5e-2]},
        metric_name="F1",
        seed=12,
    )
    result = search.run(train_ds, val_ds, batch_size=8, epochs=1)
    assert result.metric_name == "f1"
    assert result.best_trial.best_score > 0.0

    # Non-finite score ranking verification
    assert not search._is_better(float("nan"), 0.5)
    assert not search._is_better(float("inf"), 0.5)
    assert search._is_better(0.5, float("nan"))
    assert search._is_better(0.5, None)


def test_virtual_grid_search_amp_dispatch_smoke() -> None:
    """Verify VirtualGridSearch runs with bfloat16 and float16 AMP modes."""
    base_model, train_ds, val_ds = _make_synthetic_classification_data(seed=34)
    grid = VirtualGrid(world_size=2, total_memory_bytes=2 * 1024**3, device_type="cpu", seed=34)

    search_bf16 = VirtualGridSearch(
        base_model=base_model,
        grid=grid,
        param_grid={"lr": [1e-3, 5e-3]},
        amp_dtype="bfloat16",
        seed=34,
    )
    res_bf16 = search_bf16.run(train_ds, val_ds, batch_size=8, epochs=1, max_steps=1)
    assert len(res_bf16.trials) == 2

    search_fp16 = VirtualGridSearch(
        base_model=base_model,
        grid=grid,
        param_grid={"lr": [1e-3, 5e-3]},
        amp_dtype="float16",
        seed=34,
    )
    res_fp16 = search_fp16.run(train_ds, val_ds, batch_size=8, epochs=1, max_steps=1)
    assert len(res_fp16.trials) == 2


def test_virtual_grid_search_base_model_unmutated() -> None:
    """Verify that base_model weights remain completely unmutated after search completes."""
    base_model, train_ds, val_ds = _make_synthetic_classification_data(seed=45)
    orig_params = {k: v.clone() for k, v in base_model.state_dict().items()}

    grid = VirtualGrid(world_size=2, total_memory_bytes=2 * 1024**3, device_type="cpu", seed=45)
    search = VirtualGridSearch(
        base_model=base_model,
        grid=grid,
        param_grid={"lr": [1e-2, 5e-2]},
        seed=45,
    )
    search.run(train_ds, val_ds, batch_size=8, epochs=1, max_steps=2)

    for k, v in base_model.state_dict().items():
        assert torch.equal(v, orig_params[k])
