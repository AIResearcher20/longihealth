"""
Patient-level splitting for LongiHealth.

Splits admissions into train/validation/test so that no patient
appears in more than one split. Stratification uses a per-patient
binary outcome (1 if any admission was positive).
"""

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def patient_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient with a binary outcome."""
    return df.groupby("subject_id", as_index=False)["outcome"].max()


def split_patient_level(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (train, val, test), split at the patient level.

    val_size and test_size are interpreted as fractions of the full
    cohort. The first cut holds out (val + test) together; the
    second cut divides that temp set into val and test.
    """
    seed = config["project"]["random_seed"]
    val_size = config["split"]["val_size"]
    test_size = config["split"]["test_size"]

    if config["split"]["stratify_by"] != "outcome":
        raise ValueError("Only stratification by outcome is supported.")

    patients = patient_outcomes(df)

    train_patients, temp_patients = train_test_split(
        patients,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=patients["outcome"],
    )

    val_share = val_size / (val_size + test_size)
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=1.0 - val_share,
        random_state=seed + 1,
        stratify=temp_patients["outcome"],
    )

    train_ids = set(train_patients["subject_id"])
    val_ids = set(val_patients["subject_id"])
    test_ids = set(test_patients["subject_id"])

    if train_ids & val_ids:
        raise RuntimeError("Patient overlap between train and validation.")
    if train_ids & test_ids:
        raise RuntimeError("Patient overlap between train and test.")
    if val_ids & test_ids:
        raise RuntimeError("Patient overlap between validation and test.")

    train = df[df["subject_id"].isin(train_ids)].copy()
    val = df[df["subject_id"].isin(val_ids)].copy()
    test = df[df["subject_id"].isin(test_ids)].copy()

    return train, val, test


def split_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """Per-split counts of patients, admissions, and positive outcomes."""

    def stats(split: pd.DataFrame) -> dict[str, int]:
        return {
            "patients": int(split["subject_id"].nunique()),
            "admissions": len(split),
            "positive_outcomes": int(split["outcome"].sum()),
        }

    return {
        "train": stats(train),
        "validation": stats(val),
        "test": stats(test),
    }
