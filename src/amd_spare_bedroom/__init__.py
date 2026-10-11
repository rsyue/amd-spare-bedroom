"""AMD Spare Bedroom.

Emulating multiple AMD GPU devices and distributed ranks (DDP, FSDP)
within unified high-capacity VRAM on AMD ROCm (gfx_1151 / Strix Halo).
"""

from amd_spare_bedroom.communicator import VirtualCommunicator
from amd_spare_bedroom.ddp import VirtualDDP
from amd_spare_bedroom.grid import VirtualDevice, VirtualGrid
from amd_spare_bedroom.sampler import VirtualRankSampler

__version__ = "0.1.0"

__all__ = [
    "VirtualCommunicator",
    "VirtualDDP",
    "VirtualDevice",
    "VirtualGrid",
    "VirtualRankSampler",
]
