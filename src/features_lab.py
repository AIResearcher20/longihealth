"""
LongiHealth — Laboratory feature engineering.

Extracts laboratory features within the first N hours of ICU
admission, strictly bounded by feature_window_start and
feature_window_end.

Leakage rules
-------------
- Only charttime in [index_date, index_date + N hours) is used.
- ref_range_lower / ref_range_upper define abnormal_flag.
- Output: per-admission x itemid aggregation table AND a
  wide admission-level matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .data import load_dictionary


# ------------------------------------------------------------
# Aggregation statistics
# ------------------------------------------------------------

LAB_STATS = [
    "lab_count",
    "lab_mean",
    "lab_min",
    "lab_max",
    "lab_std",
    "abnormal_count",
]


def _abnormal_flag(
    df: pd.DataFrame
) -> pd.Series:
    """
    Compute a 0/1 abnormality flag based on reference ranges.
    NaN when reference ranges are unavailable.
    """

    v = df["valuenum"]
    lo = df["ref_range_lower"]
    hi = df["ref_range_upper"]

    flag = pd.Series(np.nan, index=df.index)

    has_lo = lo.notna()
    has_hi = hi.notna()

    flag[has_lo & (v < lo)] = 1
    flag[has_hi & (v > hi)] = 1
    flag[has_lo & has_hi & (v >= lo) & (v <= hi)] = 0

    return flag


def extract_lab_features(
    config: Dict[str, Any],
    cohort: pd.DataFrame,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """
    Extract per-admission x itemid laboratory features.

    Reads labevents in chunks to keep memory usage bounded.
    """

    data_root = config["paths"]["data_root"]
    version = config["mimic"]["version"]
    lab_path = (
        Path(data_root)
        / "physionet.org" / "files" / "mimic-iv-demo"
        / version
        / config["tables"]["labevents"]
    )

    if not lab_path.exists():
        raise FileNotFoundError(lab_path)

    cohort_keys = cohort[
        ["subject_id", "hadm_id",
         "index_date", "feature_window_end"]
    ].copy()

    usecols = [
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "valuenum",
        "ref_range_lower",
        "ref_range_upper",
    ]

    matched: List[pd.DataFrame] = []

    for chunk in pd.read_csv(
        lab_path,
        usecols=usecols,
        parse_dates=["charttime"],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk = chunk.merge(
            cohort_keys,
            on=["subject_id", "hadm_id"],
            how="inner",
        )

        if chunk.empty:
            continue

        chunk = chunk[
            (chunk["charttime"] >= chunk["index_date"])
            & (chunk["charttime"] < chunk["feature_window_end"])
        ]

        if chunk.empty:
            continue

        chunk = chunk[
            chunk["valuenum"].notna()
        ].copy()

        if chunk.empty:
            continue

        chunk["abnormal_flag"] = _abnormal_flag(chunk)

        matched.append(chunk)

    if not matched:
        raise RuntimeError(
            "No laboratory events found in any cohort window."
        )

    labs = pd.concat(matched, ignore_index=True)

    # --------------------------------------------------------
    # Aggregate per admission x itemid
    # --------------------------------------------------------

    grouped = (
        labs.groupby(
            ["subject_id", "hadm_id", "itemid"],
            as_index=False,
        )
        .agg(
            lab_count=("valuenum", "count"),
            lab_mean=("valuenum", "mean"),
            lab_min=("valuenum", "min"),
            lab_max=("valuenum", "max"),
            lab_std=("valuenum", "std"),
            abnormal_count=("abnormal_flag", "sum"),
        )
    )

    grouped["lab_std"] = grouped["lab_std"].fillna(0)

    return grouped


def build_lab_matrix(
    lab_features: pd.DataFrame,
    cohort: pd.DataFrame,
    min_observed_fraction: float,
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivot per-item lab features into a wide admission-level
    matrix, filtered by observed fraction on the whole cohort.

    Note: this function does NOT filter using outcome or
    labels; only observed prevalence.
    """

    # --------------------------------------------------------
    # Pivot each statistic into its own wide table
    # --------------------------------------------------------

    stats = [
        "lab_mean",
        "lab_min",
        "lab_max",
        "lab_std",
        "lab_count",
        "abnormal_count",
    ]

    matrices = []

    for stat in stats:

        wide = (
            lab_features
            .pivot_table(
                index=["subject_id", "hadm_id"],
                columns="itemid",
                values=stat,
                aggfunc="mean",
            )
        )

        wide.columns = [
            f"{stat}_{int(c)}"
            for c in wide.columns
        ]

        matrices.append(wide)

    lab_wide = pd.concat(matrices, axis=1).reset_index()

    # --------------------------------------------------------
    # Filter by observed fraction (whole cohort prevalence)
    # --------------------------------------------------------

    feature_cols = [
        c for c in lab_wide.columns
        if c not in ("subject_id", "hadm_id")
    ]

    n = len(lab_wide)
    observed = lab_wide[feature_cols].notna().sum() / n
    nunique = lab_wide[feature_cols].nunique(dropna=True)

    keep = [
        c for c in feature_cols
        if observed[c] >= min_observed_fraction
        and nunique[c] > 1
    ]

    lab_wide = lab_wide[
        ["subject_id", "hadm_id"] + keep
    ]

    # --------------------------------------------------------
    # Rename to readable labels
    # --------------------------------------------------------

    def _safe_label(s: str) -> str:
        return (
            str(s).lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace("(", "").replace(")", "")
            .replace("%", "pct")
        )

    id_to_label = dict(
        zip(dictionary["itemid"], dictionary["label"])
    )

    rename = {}
    for c in keep:
        prefix, itemid_str = c.rsplit("_", 1)
        try:
            itemid = int(itemid_str)
        except ValueError:
            continue
        label = id_to_label.get(itemid)
        if label:
            rename[c] = f"{prefix}_{_safe_label(label)}_{itemid}"

    lab_wide = lab_wide.rename(columns=rename)

    # --------------------------------------------------------
    # Merge with cohort
    # --------------------------------------------------------

    final = cohort.merge(
        lab_wide,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )

    return final
