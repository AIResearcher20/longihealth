"""
LongiHealth — Patient-level dataset splitting.

Prevents the same patient from appearing in more than one split.

Steps
-----
1. Build a patient-level outcome table (max of admission outcomes).
2. Stratified split into train / temp.
3. Stratified split of temp into validation / test.
4. Assign whole patients (all their admissions) to their split.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def _patient_outcomes(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a patient-level outcome table.

    A patient is positive if ANY of their admissions is positive.
    """

    return (
        df.groupby("subject_id")["outcome"]
        .max()
        .reset_index()
    )


def split_patient_level(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train / validation / test at the
    patient level.

    Returns
    -------
    (train, validation, test)
    """

    seed = config["project"]["random_seed"]
    test_size = config["split"]["test_size"]
    val_size = config["split"]["val_size"]
    stratify_by = config["split"]["stratify_by"]

    if stratify_by != "outcome":
        raise ValueError(
            "Only stratification by outcome is supported."
        )

    patients = _patient_outcomes(df)

    # --------------------------------------------------------
    # 1. Train vs temporary
    # --------------------------------------------------------

    train_patients, temp_patients = train_test_split(
        patients,
        test_size=test_size,
        random_state=seed,
        stratify=patients["outcome"],
    )

    # --------------------------------------------------------
    # 2. Validation vs test
    # --------------------------------------------------------

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=val_size,
        random_state=seed,
        stratify=temp_patients["outcome"],
    )

    train_ids = set(train_patients["subject_id"])
    val_ids = set(val_patients["subject_id"])
    test_ids = set(test_patients["subject_id"])

    # --------------------------------------------------------
    # 3. Sanity: no overlap
    # --------------------------------------------------------

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # --------------------------------------------------------
    # 4. Materialise splits
    # --------------------------------------------------------

    train = df[df["subject_id"].isin(train_ids)].copy()
    val = df[df["subject_id"].isin(val_ids)].copy()
    test = df[df["subject_id"].isin(test_ids)].copy()

    return train, val, test


def split_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """
    Compact summary of a patient-level split.
    """

    def _stats(df: pd.DataFrame) -> Dict[str, int]:
        return {
            "patients": int(df["subject_id"].nunique()),
            "admissions": int(len(df)),
            "positive_outcomes": int(df["outcome"].sum()),
        }

    return {
        "train": _stats(train),
        "validation": _stats(val),
        "test": _stats(test),
    }
