"""AMD Spare Bedroom.

Emulating multiple AMD GPU devices and distributed ranks (DDP, FSDP)
within unified high-capacity VRAM on AMD ROCm (gfx_1151 / Strix Halo).
"""

from amd_spare_bedroom.amp import AmpGradScalerState, resolve_amp_dtype
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
    HyperparameterGrid,
    TrialConfig,
    WaveSchedule,
)
from amd_spare_bedroom.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    evaluate_classification_epoch,
)
from amd_spare_bedroom.sampler import VirtualRankSampler

__version__ = "0.1.0"
__license__ = "CC-BY-NC-ND-4.0"

__all__ = [
    "AmpGradScalerState",
    "BestCheckpointSaver",
    "ClassificationMetrics",
    "HyperparameterGrid",
    "TrialConfig",
    "VirtualCommunicator",
    "VirtualDDP",
    "VirtualDevice",
    "VirtualFSDP",
    "VirtualGrid",
    "VirtualRankSampler",
    "WaveSchedule",
    "compute_classification_metrics",
    "evaluate_classification_epoch",
    "resolve_amp_dtype",
    "seed_everything",
]
