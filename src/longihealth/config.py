"""
LongiHealth — Configuration loader.

Loads and validates the project YAML configuration.
All pipeline parameters live in `configs/default.yaml`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(
    path: str | Path | None = None
) -> Dict[str, Any]:
    """
    Load the LongiHealth YAML configuration.

    Parameters
    ----------
    path : str | Path | None
        Path to the YAML config. If None, uses
        `configs/default.yaml`.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _validate_config(config)

    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Minimal structural validation of the config file.
    """

    required_sections = [
        "project",
        "paths",
        "mimic",
        "tables",
        "cohort",
        "features",
        "split",
        "preprocessing",
        "models",
        "evaluation",
    ]

    for section in required_sections:
        if section not in config:
            raise KeyError(
                f"Config missing required section: {section}"
            )

    window = config["cohort"].get("feature_window_hours")

    if not isinstance(window, int) or window <= 0:
        raise ValueError(
            "cohort.feature_window_hours must be a positive integer."
        )

    if config["split"]["level"] != "patient":
        raise ValueError(
            "Only patient-level splitting is supported."
        )


def resolve_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Convert path strings from config into Path objects.
    """

    return {
        key: Path(value)
        for key, value in config["paths"].items()
    }
