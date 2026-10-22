"""Environment configuration, .env file parsing, and runtime environment variable abstraction.

Provides automatic discovery and parsing of `.env` files without external dependencies,
supporting standard syntax (comments, quotes, export prefixes), non-destructive population
of `os.environ`, and typed getter helpers with fallbacks.
"""

from __future__ import annotations

import os
from pathlib import Path


def parse_dotenv_content(content: str) -> dict[str, str]:
    """Parse raw .env content into a dictionary of string key-value pairs.

    Supports:
    - Empty lines and comments starting with '#'
    - Optional 'export ' prefix
    - Single and double quoted values
    - Inline comments after unquoted or quoted values
    """
    env_vars: dict[str, str] = {}
    for line in content.splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue

        if trimmed.startswith("export "):
            trimmed = trimmed[len("export ") :].strip()

        if "=" not in trimmed:
            continue

        key, raw_val = trimmed.split("=", 1)
        key = key.strip()
        if not key:
            continue

        val = raw_val.strip()

        if val.startswith('"'):
            # Double-quoted value: find matching closing quote
            end_idx = val.find('"', 1)
            if end_idx != -1:
                val = val[1:end_idx]
            else:
                val = val[1:]
        elif val.startswith("'"):
            # Single-quoted value: find matching closing quote
            end_idx = val.find("'", 1)
            if end_idx != -1:
                val = val[1:end_idx]
            else:
                val = val[1:]
        else:
            # Unquoted: strip trailing comment if any
            if " #" in val:
                val = val.split(" #", 1)[0].rstrip()
            elif "\t#" in val:
                val = val.split("\t#", 1)[0].rstrip()

        env_vars[key] = val

    return env_vars


def find_dotenv(
    filename: str = ".env",
    start_dir: str | Path | None = None,
) -> Path | None:
    """Search for a .env file starting at start_dir and walking up parent directories.

    Stops walking upward when encountering repository boundary markers (.git or pyproject.toml)
    or the filesystem root.
    """
    filename_path = Path(filename)
    if filename_path.is_absolute() or len(filename_path.parts) > 1:
        raise ValueError(f"filename must be a plain filename, got: {filename!r}")

    current = Path(start_dir).resolve() if start_dir is not None else Path.cwd().resolve()
    for parent in [current, *current.parents]:
        target = parent / filename
        if target.is_file():
            return target
        if (parent / ".git").exists() or (parent / "pyproject.toml").is_file():
            break
    return None


def load_dotenv(
    path: str | Path | None = None,
    override: bool = False,
) -> dict[str, str]:
    """Discover, parse, and load environment variables from a .env file into os.environ.

    Args:
        path: Explicit path to .env file, or None to auto-discover by walking upward.
        override: Whether to overwrite variables that already exist in os.environ.

    Returns:
        Dictionary of parsed key-value pairs from the .env file.
    """
    target_path = Path(path).resolve() if path is not None else find_dotenv()
    if target_path is None or not target_path.is_file():
        return {}

    try:
        content = target_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    parsed = parse_dotenv_content(content)
    for key, val in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = val

    return parsed


def get_env_str(name: str, default: str = "") -> str:
    """Retrieve an environment variable as a string, with fallback default."""
    return os.environ.get(name, default)


def get_env_int(name: str, default: int = 0) -> int:
    """Retrieve an environment variable as an integer, raising ValueError if invalid."""
    val = os.environ.get(name)
    if val is None or not val.strip():
        return default
    try:
        return int(val.strip())
    except ValueError:
        raise ValueError(
            f"Environment variable '{name}' with value '{val}' cannot be parsed as int."
        ) from None


def get_env_float(name: str, default: float = 0.0) -> float:
    """Retrieve an environment variable as a float, raising ValueError if invalid."""
    val = os.environ.get(name)
    if val is None or not val.strip():
        return default
    try:
        return float(val.strip())
    except ValueError:
        raise ValueError(
            f"Environment variable '{name}' with value '{val}' cannot be parsed as float."
        ) from None


def get_env_bool(name: str, default: bool = False) -> bool:
    """Retrieve an environment variable as a boolean."""
    val = os.environ.get(name)
    if val is None or not val.strip():
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")
