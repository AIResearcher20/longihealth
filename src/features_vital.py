"""
LongiHealth — Vital-sign feature engineering.

Extracts vital-sign features within the first N hours of ICU
admission using an explicit whitelist of itemids.

Leakage rules
-------------
- Only charttime in [index_date, index_date + N hours).
- Only item IDs explicitly listed in config
  (features.vital.selected_item_ids).
- Aggregation only; no outcome or post-outcome variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .data import load_dictionary


# ------------------------------------------------------------
# Aggregation statistics
# ------------------------------------------------------------

VITAL_STATS = [
    "vital_count",
    "vital_mean",
    "vital_min",
    "vital_max",
    "vital_std",
]


def extract_vital_features(
    config: Dict[str, Any],
    cohort: pd.DataFrame,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """
    Extract per-admission x itemid vital-sign features.
    """

    data_root = config["paths"]["data_root"]
    version = config["mimic"]["version"]

    chart_path = (
        Path(data_root)
        / "physionet.org" / "files" / "mimic-iv-demo"
        / version
        / config["tables"]["chartevents"]
    )

    if not chart_path.exists():
        raise FileNotFoundError(chart_path)

    selected_item_ids = set(
        config["features"]["vital"]["selected_item_ids"]
    )

    cohort_keys = cohort[
        ["subject_id", "hadm_id", "stay_id",
         "index_date", "feature_window_end"]
    ].copy()

    usecols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "charttime",
        "itemid",
        "value",
        "valuenum",
    ]

    matched: List[pd.DataFrame] = []

    for chunk in pd.read_csv(
        chart_path,
        usecols=usecols,
        parse_dates=["charttime"],
        chunksize=chunksize,
        low_memory=False,
    ):

        # Keep only cohort stays
        chunk = chunk[
            chunk["stay_id"].isin(cohort_keys["stay_id"])
        ]

        if chunk.empty:
            continue

        # Keep only whitelisted vital itemids
        chunk = chunk[
            chunk["itemid"].isin(selected_item_ids)
        ]

        if chunk.empty:
            continue

        # Merge window boundaries
        chunk = chunk.merge(
            cohort_keys,
            on="stay_id",
            how="inner",
            suffixes=("", "_cohort"),
        )

        # Strict 24-hour window
        chunk = chunk[
            (chunk["charttime"] >= chunk["index_date"])
            & (chunk["charttime"] < chunk["feature_window_end"])
        ]

        if chunk.empty:
            continue

        # Numeric only
        chunk["valuenum"] = pd.to_numeric(
            chunk["valuenum"], errors="coerce"
        )
        chunk = chunk[chunk["valuenum"].notna()]

        if chunk.empty:
            continue

        matched.append(
            chunk[
                ["subject_id", "hadm_id", "stay_id",
                 "charttime", "itemid", "valuenum"]
            ]
        )

    if not matched:
        raise RuntimeError(
            "No vital-sign events found in any cohort window."
        )

    vitals = pd.concat(matched, ignore_index=True)

    # --------------------------------------------------------
    # Aggregate per admission x itemid
    # --------------------------------------------------------

    grouped = (
        vitals.groupby(
            ["subject_id", "hadm_id", "itemid"],
            as_index=False,
        )
        .agg(
            vital_count=("valuenum", "count"),
            vital_mean=("valuenum", "mean"),
            vital_min=("valuenum", "min"),
            vital_max=("valuenum", "max"),
            vital_std=("valuenum", "std"),
        )
    )

    grouped["vital_std"] = grouped["vital_std"].fillna(0)

    return grouped


def build_vital_matrix(
    vital_features: pd.DataFrame,
    cohort: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivot per-item vital features into a wide admission-level
    matrix and merge with the cohort.
    """

    # --------------------------------------------------------
    # Attach readable labels
    # --------------------------------------------------------

    labels = (
        dictionary[["itemid", "label"]]
        .drop_duplicates("itemid")
    )

    vital_features = vital_features.merge(
        labels, on="itemid", how="left"
    )

    # --------------------------------------------------------
    # Pivot each statistic
    # --------------------------------------------------------

    matrices = []

    for stat in VITAL_STATS:

        wide = vital_features.pivot_table(
            index=["subject_id", "hadm_id"],
            columns="itemid",
            values=stat,
            aggfunc="mean",
        )

        # Label-based column names
        itemid_to_label = dict(
            zip(
                vital_features["itemid"],
                vital_features["label"].fillna("unknown"),
            )
        )

        def _safe(s: str) -> str:
            return (
                str(s).lower()
                .replace(" ", "_")
                .replace("/", "_")
                .replace("-", "_")
                .replace("(", "").replace(")", "")
                .replace("%", "pct")
            )

        wide.columns = [
            f"{stat}_{_safe(itemid_to_label.get(int(c), 'unknown'))}_{int(c)}"
            for c in wide.columns
        ]

        matrices.append(wide)

    vital_wide = pd.concat(matrices, axis=1).reset_index()

    # --------------------------------------------------------
    # Merge with cohort
    # --------------------------------------------------------

    final = cohort.merge(
        vital_wide,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )

    return final
