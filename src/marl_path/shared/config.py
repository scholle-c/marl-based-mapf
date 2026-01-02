"""Config loading helpers for the CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for older versions
    tomllib = None  # type: ignore


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a configuration file (JSON, TOML, or YAML if PyYAML is installed).

    Parameters
    ----------
    path:
        Path to the config file.

    Returns
    -------
    dict
        Parsed configuration as a dictionary.
    """
    config_path = Path(path)
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if suffix == ".toml":
        if tomllib is None:
            raise ImportError(
                "tomllib is required to read TOML configs (Python 3.11+)."
            )
        with config_path.open("rb") as f:
            return tomllib.load(f)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ImportError("PyYAML is required to read YAML configs.") from exc
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    raise ValueError(
        f"Unsupported config format for {config_path}. Use JSON, TOML, or YAML."
    )
