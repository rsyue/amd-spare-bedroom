"""Unit tests for .env discovery, parsing, and environment variable abstraction."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from amd_spare_bedroom.config.env import (
    find_dotenv,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_str,
    load_dotenv,
    parse_dotenv_content,
)


def test_parse_dotenv_content_basic_and_quotes() -> None:
    raw = """
    # Comment line
    SIMPLE=value
    EXPORT_VAR=export_val
    SINGLE_QUOTED='single value'
    DOUBLE_QUOTED="double value"
    SPACED = spaced_value
    TRAILING_COMMENT=foo # this is a comment
    QUOTED_WITH_HASH="hello # world"
    EMPTY_VAL=
    """
    parsed = parse_dotenv_content(raw)
    assert parsed["SIMPLE"] == "value"
    assert parsed["EXPORT_VAR"] == "export_val"
    assert parsed["SINGLE_QUOTED"] == "single value"
    assert parsed["DOUBLE_QUOTED"] == "double value"
    assert parsed["SPACED"] == "spaced_value"
    assert parsed["TRAILING_COMMENT"] == "foo"
    assert parsed["QUOTED_WITH_HASH"] == "hello # world"
    assert parsed["EMPTY_VAL"] == ""


def test_parse_dotenv_content_export_prefix() -> None:
    raw = """
    export ROCM_PATH=/opt/rocm
    export HSA_OVERRIDE_GFX_VERSION="11.5.1"
    """
    parsed = parse_dotenv_content(raw)
    assert parsed["ROCM_PATH"] == "/opt/rocm"
    assert parsed["HSA_OVERRIDE_GFX_VERSION"] == "11.5.1"


def test_find_dotenv_in_dir(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_KEY=123\n", encoding="utf-8")

    sub_dir = tmp_path / "sub" / "deep"
    sub_dir.mkdir(parents=True)

    found = find_dotenv(start_dir=sub_dir)
    assert found is not None
    assert found.resolve() == env_file.resolve()


def test_find_dotenv_not_found(tmp_path: Path) -> None:
    found = find_dotenv(start_dir=tmp_path)
    assert found is None


def test_load_dotenv_no_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXISTING_KEY", "initial_val")
    monkeypatch.delenv("NEW_KEY", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_KEY=new_val\nNEW_KEY=hello\n", encoding="utf-8")

    loaded = load_dotenv(path=env_file, override=False)
    assert loaded["EXISTING_KEY"] == "new_val"
    assert loaded["NEW_KEY"] == "hello"

    # os.environ should preserve existing value
    assert os.environ["EXISTING_KEY"] == "initial_val"
    assert os.environ["NEW_KEY"] == "hello"


def test_load_dotenv_with_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXISTING_KEY", "initial_val")
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_KEY=overridden_val\n", encoding="utf-8")

    load_dotenv(path=env_file, override=True)
    assert os.environ["EXISTING_KEY"] == "overridden_val"


def test_load_dotenv_missing_file_returns_empty() -> None:
    loaded = load_dotenv(path="/non/existent/path/.env")
    assert loaded == {}


def test_typed_getters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STR_VAR", "hello")
    monkeypatch.setenv("INT_VAR", "42")
    monkeypatch.setenv("FLOAT_VAR", "3.14")
    monkeypatch.setenv("BOOL_TRUE_1", "1")
    monkeypatch.setenv("BOOL_TRUE_YES", "true")
    monkeypatch.setenv("BOOL_FALSE", "0")

    assert get_env_str("STR_VAR", "default") == "hello"
    assert get_env_str("MISSING_VAR", "default") == "default"

    assert get_env_int("INT_VAR", 0) == 42
    assert get_env_int("MISSING_INT", 10) == 10

    assert get_env_float("FLOAT_VAR", 0.0) == 3.14
    assert get_env_float("MISSING_FLOAT", 1.5) == 1.5

    assert get_env_bool("BOOL_TRUE_1", False) is True
    assert get_env_bool("BOOL_TRUE_YES", False) is True
    assert get_env_bool("BOOL_FALSE", True) is False
    assert get_env_bool("MISSING_BOOL", True) is True


def test_typed_getters_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAD_INT", "not_an_int")
    monkeypatch.setenv("BAD_FLOAT", "not_a_float")

    with pytest.raises(ValueError, match="cannot be parsed as int"):
        get_env_int("BAD_INT")

    with pytest.raises(ValueError, match="cannot be parsed as float"):
        get_env_float("BAD_FLOAT")


def test_find_dotenv_invalid_filename() -> None:
    with pytest.raises(ValueError, match="plain filename"):
        find_dotenv(filename="/etc/shadow")

    with pytest.raises(ValueError, match="plain filename"):
        find_dotenv(filename="../secret.env")


def test_find_dotenv_stops_at_repo_boundary(tmp_path: Path) -> None:
    outer_env = tmp_path / ".env"
    outer_env.write_text("OUTER=true\n", encoding="utf-8")

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    deep_dir = repo_root / "src" / "pkg"
    deep_dir.mkdir(parents=True)

    assert find_dotenv(start_dir=deep_dir) is None


def test_get_env_bool_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMPTY_BOOL", "")
    monkeypatch.setenv("WHITESPACE_BOOL", "   ")

    assert get_env_bool("EMPTY_BOOL", default=True) is True
    assert get_env_bool("WHITESPACE_BOOL", default=True) is True
