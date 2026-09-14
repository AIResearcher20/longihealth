"""
LongiHealth — Final clinical feature matrix.

Merges:
    cohort  (identifiers, index date, outcome)
    lab     (wide laboratory matrix)
    vital   (wide vital-sign matrix)

Design principles
-----------------
- No outcome or post-outcome column is ever a feature.
- Identifiers and timestamps are preserved but never used
  as model inputs.
- One row per hospital admission.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


# ------------------------------------------------------------
# Columns that are NEVER features
# ------------------------------------------------------------

NON_FEATURE_COLUMNS: List[str] = [
    # identifiers
    "subject_id",
    "hadm_id",
    "stay_id",

    # timestamps
    "index_date",
    "feature_window_start",
    "feature_window_end",
    "intime",
    "outtime",

    # administrative / discharge
    "admission_type",
    "admit_provider_id",
    "admission_location",
    "discharge_location",
    "insurance",
    "language",
    "marital_status",
    "race",
    "first_careunit",
    "last_careunit",

    # demographics (kept as metadata)
    "gender",
    "anchor_age",
    "anchor_year_group",

    # OUTCOME and post-outcome variables
    "hospital_expire_flag",
    "outcome",
    "los",
    "deathtime",
    "dischtime",
]


def merge_clinical_features(
    cohort: pd.DataFrame,
    lab_matrix: pd.DataFrame,
    vital_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge cohort + lab matrix + vital matrix into one table.

    All three DataFrames must have unique (subject_id, hadm_id).
    """

    keys = ["subject_id", "hadm_id"]

    # --------------------------------------------------------
    # 1. Identify lab & vital feature columns
    # --------------------------------------------------------

    lab_feature_cols = [
        c for c in lab_matrix.columns
        if c not in NON_FEATURE_COLUMNS
        and c not in keys
    ]

    vital_feature_cols = [
        c for c in vital_matrix.columns
        if c not in NON_FEATURE_COLUMNS
        and c not in keys
    ]

    # --------------------------------------------------------
    # 2. Keep only keys + features from each matrix
    # --------------------------------------------------------

    labs = lab_matrix[keys + lab_feature_cols].copy()
    vitals = vital_matrix[keys + vital_feature_cols].copy()

    # --------------------------------------------------------
    # 3. Cohort metadata we want to keep alongside features
    # --------------------------------------------------------

    metadata_cols = [
        c for c in [
            "subject_id",
            "hadm_id",
            "stay_id",
            "index_date",
            "outcome",
        ]
        if c in cohort.columns
    ]

    base = cohort[metadata_cols].copy()

    # --------------------------------------------------------
    # 4. Merge
    # --------------------------------------------------------

    final = base.merge(
        labs,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    final = final.merge(
        vitals,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # 5. Leakage assertions
    # --------------------------------------------------------

    forbidden_present = [
        c for c in final.columns
        if c in {
            "hospital_expire_flag",
            "los",
            "deathtime",
            "dischtime",
        }
    ]

    if forbidden_present:
        raise RuntimeError(
            "Leakage columns found in final matrix: "
            f"{forbidden_present}"
        )

    return final


def list_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return only the feature columns from the final matrix.
    """

    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLUMNS
    ]


def feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Short summary of the final feature matrix.
    """

    lab_feats = [
        c for c in df.columns
        if c.startswith(
            ("lab_mean_", "lab_min_", "lab_max_",
             "lab_std_", "lab_count_", "abnormal_count_")
        )
    ]

    vital_feats = [
        c for c in df.columns
        if c.startswith(
            ("vital_mean_", "vital_min_", "vital_max_",
             "vital_std_", "vital_count_")
        )
    ]

    all_feats = lab_feats + vital_feats

    missingness = (
        df[all_feats].isna().mean().mean()
        if all_feats else 0.0
    )

    return {
        "rows": int(len(df)),
        "unique_patients": int(df["subject_id"].nunique()),
        "unique_admissions": int(df["hadm_id"].nunique()),
        "lab_features": len(lab_feats),
        "vital_features": len(vital_feats),
        "total_features": len(all_feats),
        "positive_outcomes": int(df["outcome"].sum()),
        "overall_missingness": float(missingness),
    }
