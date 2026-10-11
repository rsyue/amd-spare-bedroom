"""Test basic package metadata and environment scaffolding."""

import amd_spare_bedroom


def test_package_version() -> None:
    """Verify package version is set."""
    assert amd_spare_bedroom.__version__ == "0.1.0"


def test_package_license() -> None:
    """Verify package license is set to CC-BY-NC-ND-4.0."""
    assert amd_spare_bedroom.__license__ == "CC-BY-NC-ND-4.0"
