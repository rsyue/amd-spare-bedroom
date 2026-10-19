"""AMD Spare Bedroom.

Emulating multiple AMD GPU devices and distributed ranks (DDP, FSDP)
within unified high-capacity VRAM on AMD ROCm (gfx_1151 / Strix Halo).
"""

from amd_spare_bedroom.amp import (
    ROCM_GRAD_FINITE_BOUND,
    AmpGradScalerState,
    resolve_amp_dtype,
)
from amd_spare_bedroom.checkpoint import BestCheckpointSaver
from amd_spare_bedroom.communicator import VirtualCommunicator
from amd_spare_bedroom.ddp import VirtualDDP
from amd_spare_bedroom.fsdp import VirtualFSDP
from amd_spare_bedroom.grid import (
    VirtualDevice,
    VirtualGrid,
    seed_everything,
)
from amd_spare_bedroom.grid_search import (
    DEFAULT_LR_GRID_STR,
    DEFAULT_WEIGHT_DECAY_GRID_STR,
    GridSearchResult,
    HyperparameterGrid,
    TrialConfig,
    TrialResult,
    VirtualGridSearch,
    WaveSchedule,
    parse_float_grid,
)
from amd_spare_bedroom.metrics import (
    ClassificationMetrics,
    SFTMetrics,
    compute_classification_metrics,
    compute_sft_metrics,
    evaluate_causal_lm_epoch,
    evaluate_causal_lm_model,
    evaluate_classification_epoch,
)
from amd_spare_bedroom.sampler import VirtualRankSampler

__version__ = "0.1.0"
__license__ = "CC-BY-NC-ND-4.0"

__all__ = [
    "AmpGradScalerState",
    "BestCheckpointSaver",
    "ClassificationMetrics",
    "DEFAULT_LR_GRID_STR",
    "DEFAULT_WEIGHT_DECAY_GRID_STR",
    "GridSearchResult",
    "HyperparameterGrid",
    "ROCM_GRAD_FINITE_BOUND",
    "SFTMetrics",
    "TrialConfig",
    "TrialResult",
    "VirtualCommunicator",
    "VirtualDDP",
    "VirtualDevice",
    "VirtualFSDP",
    "VirtualGrid",
    "VirtualGridSearch",
    "VirtualRankSampler",
    "WaveSchedule",
    "compute_classification_metrics",
    "compute_sft_metrics",
    "evaluate_causal_lm_epoch",
    "evaluate_causal_lm_model",
    "evaluate_classification_epoch",
    "parse_float_grid",
    "resolve_amp_dtype",
    "seed_everything",
]
