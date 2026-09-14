"""
LongiHealth — Cohort construction.

Builds an admission-level ICU cohort from MIMIC-IV Demo.

Design principles
------------------
- Index event: FIRST ICU stay per hospital admission
- Feature window: first N hours after ICU intime
- Outcome: hospital mortality (OUTCOME ONLY — never a feature)
- No leakage: length-of-stay and discharge/death times are not
  used as features.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .data import load_core_tables


# ------------------------------------------------------------
# Column selection for the final cohort
# ------------------------------------------------------------

COHORT_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "intime",
    "outtime",
    "first_careunit",
    "last_careunit",
    "los",
    "index_date",
    "feature_window_start",
    "feature_window_end",
    "gender",
    "anchor_age",
    "hospital_expire_flag",
    "outcome",
]


def build_cohort(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Build the admission-level ICU cohort.

    Steps
    -----
    1. Load admissions, patients, icustays.
    2. Keep the first ICU stay per hospital admission.
    3. Merge patient demographics.
    4. Define index_date = ICU intime.
    5. Define the feature window (index_date -> +N hours).
    6. Define the binary outcome.
    """

    tables = load_core_tables(config)

    admissions = tables["admissions"].copy()
    patients = tables["patients"].copy()
    icustays = tables["icustays"].copy()

    # --------------------------------------------------------
    # 1. First ICU stay per hospital admission
    # --------------------------------------------------------

    icustays = icustays.sort_values(
        ["hadm_id", "intime"]
    )

    first_icu = (
        icustays
        .drop_duplicates(subset=["hadm_id"], keep="first")
        .copy()
    )

    # --------------------------------------------------------
    # 2. Merge ICU stay with hospital admission
    # --------------------------------------------------------

    cohort = admissions.merge(
        first_icu[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "first_careunit",
                "last_careunit",
                "intime",
                "outtime",
                "los",
            ]
        ],
        on=["subject_id", "hadm_id"],
        how="inner",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # 3. Patient demographics
    # --------------------------------------------------------

    cohort = cohort.merge(
        patients[
            [
                "subject_id",
                "gender",
                "anchor_age",
                "anchor_year_group",
            ]
        ],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # 4. Index date = ICU intime
    # --------------------------------------------------------

    cohort["index_date"] = cohort["intime"]

    # --------------------------------------------------------
    # 5. Feature window
    # --------------------------------------------------------

    window_hours = config["cohort"]["feature_window_hours"]

    cohort["feature_window_start"] = cohort["index_date"]
    cohort["feature_window_end"] = (
        cohort["index_date"]
        + pd.Timedelta(hours=window_hours)
    )

    # --------------------------------------------------------
    # 6. Outcome (OUTCOME ONLY)
    # --------------------------------------------------------

    outcome_col = config["cohort"]["outcome_column"]

    if outcome_col not in cohort.columns:
        raise KeyError(
            f"Outcome column missing: {outcome_col}"
        )

    cohort["outcome"] = (
        cohort[outcome_col]
        .astype(int)
    )

    # --------------------------------------------------------
    # 7. Restrict to columns we actually use
    # --------------------------------------------------------

    keep = [
        c for c in COHORT_COLUMNS
        if c in cohort.columns
    ]

    cohort = (
        cohort[keep]
        .drop_duplicates(subset=["hadm_id"])
        .reset_index(drop=True)
    )

    return cohort


def cohort_summary(cohort: pd.DataFrame) -> Dict[str, Any]:
    """
    Basic QC summary of the cohort.
    """

    return {
        "admissions": int(len(cohort)),
        "patients": int(cohort["subject_id"].nunique()),
        "unique_hadm": int(cohort["hadm_id"].nunique()),
        "duplicate_hadm": int(cohort["hadm_id"].duplicated().sum()),
        "positive_outcomes": int(cohort["outcome"].sum()),
        "mortality_rate": float(cohort["outcome"].mean()),
    }
