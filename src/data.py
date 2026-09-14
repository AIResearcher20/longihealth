"""
LongiHealth — Data ingestion.

Downloads the MIMIC-IV Clinical Database Demo (v2.2) from
PhysioNet and provides safe loaders for the required tables.

Clinical data are never committed to version control.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict

import pandas as pd


# ------------------------------------------------------------
# Download
# ------------------------------------------------------------

def download_mimic_demo(
    base_url: str,
    data_root: str | Path,
) -> Path:
    """
    Download the MIMIC-IV Demo dataset via wget.

    Idempotent: wget -r -N -c will skip already-downloaded files.
    """

    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "wget",
        "-r", "-N", "-c", "-np", "-q",
        "--show-progress",
        "-P", str(data_root),
        base_url,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"wget failed:\n{result.stderr}"
        )

    return data_root


# ------------------------------------------------------------
# Table loaders
# ------------------------------------------------------------

def _table_path(
    data_root: Path,
    mimic_version: str,
    relative_path: str,
) -> Path:
    """
    Resolve a table path under the versioned MIMIC root.
    """

    return (
        data_root
        / "physionet.org"
        / "files"
        / "mimic-iv-demo"
        / mimic_version
        / relative_path
    )


def load_table(
    data_root: str | Path,
    mimic_version: str,
    relative_path: str,
    parse_dates: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load a single MIMIC table into a DataFrame.
    """

    path = _table_path(
        Path(data_root),
        mimic_version,
        relative_path,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing table: {path}"
        )

    return pd.read_csv(
        path,
        parse_dates=parse_dates or [],
        nrows=nrows,
        low_memory=False,
    )


def load_core_tables(
    config: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """
    Load the core tables required to build the ICU cohort.
    """

    data_root = config["paths"]["data_root"]
    version = config["mimic"]["version"]
    tables = config["tables"]

    return {
        "admissions": load_table(
            data_root, version,
            tables["admissions"],
            parse_dates=[
                "admittime", "dischtime",
                "deathtime", "edregtime", "edouttime",
            ],
        ),
        "patients": load_table(
            data_root, version,
            tables["patients"],
        ),
        "icustays": load_table(
            data_root, version,
            tables["icustays"],
            parse_dates=["intime", "outtime"],
        ),
    }


def load_labevents(
    config: Dict[str, Any],
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load labevents (optionally truncated).
    """

    return load_table(
        config["paths"]["data_root"],
        config["mimic"]["version"],
        config["tables"]["labevents"],
        parse_dates=["charttime"],
        nrows=nrows,
    )


def load_chartevents(
    config: Dict[str, Any],
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load chartevents (optionally truncated).
    """

    return load_table(
        config["paths"]["data_root"],
        config["mimic"]["version"],
        config["tables"]["chartevents"],
        parse_dates=["charttime"],
        nrows=nrows,
    )


def load_dictionary(
    config: Dict[str, Any],
    table: str,
) -> pd.DataFrame:
    """
    Load a dictionary table: d_labitems or d_items.
    """

    if table not in ("d_labitems", "d_items"):
        raise ValueError(
            f"Unknown dictionary: {table}"
        )

    return load_table(
        config["paths"]["data_root"],
        config["mimic"]["version"],
        config["tables"][table],
    )
