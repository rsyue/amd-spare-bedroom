"""Test basic package metadata and environment scaffolding."""

import amd_spare_bedroom


def test_package_version() -> None:
    """Verify package version is set."""
    assert amd_spare_bedroom.__version__ == "0.1.0"
