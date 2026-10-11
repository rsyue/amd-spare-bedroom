"""AMD Spare Bedroom.

Emulating multiple AMD GPU devices and distributed ranks (DDP, FSDP)
within unified high-capacity VRAM on AMD ROCm (gfx_1151 / Strix Halo).
"""

from amd_spare_bedroom.grid import VirtualDevice, VirtualGrid

__version__ = "0.1.0"

__all__ = ["VirtualDevice", "VirtualGrid"]
