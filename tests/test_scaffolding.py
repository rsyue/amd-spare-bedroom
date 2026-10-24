"""Test basic package metadata, environment scaffolding, and license parity."""

from __future__ import annotations

import tomllib
from pathlib import Path

import amd_spare_bedroom


def test_package_metadata_matches_pyproject() -> None:
    """Verify package version and license match pyproject.toml specification."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert pyproject_path.is_file(), "pyproject.toml must exist in repo root"

    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)

    project_table = pyproject_data.get("project", {})
    assert amd_spare_bedroom.__version__ == project_table.get("version")
    assert amd_spare_bedroom.__license__ == project_table.get("license", {}).get("text")


def test_license_file_on_disk() -> None:
    """Verify LICENSE file exists and contains the MIT grant and copyright notice."""
    license_path = Path(__file__).resolve().parent.parent / "LICENSE"
    assert license_path.is_file(), "LICENSE file must exist in repo root"
    content = license_path.read_text(encoding="utf-8")
    assert content.startswith("MIT License")
    assert "Copyright (c) 2026 Richard Yue and AMD Spare Bedroom Contributors" in content
