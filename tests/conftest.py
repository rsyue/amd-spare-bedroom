"""Pytest configuration and fixtures for amd-spare-bedroom."""

import pytest


def has_rocm_gpu() -> bool:
    """Check if ROCm / HIP GPU device is available via PyTorch."""
    try:
        import torch

        return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None)
    except Exception:
        return False


@pytest.fixture
def rocm_available() -> bool:
    """Fixture returning True if an AMD ROCm GPU is physically accessible."""
    return has_rocm_gpu()
