"""
LongiHealth — Pipeline tests.

These tests run against the real src/longihealth/ package.
They do not require a full MIMIC download; they exercise the
structural and leakage-safety guarantees of the pipeline.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


from longihealth.config import load_config
from longihealth.cohort import cohort_summary
from longihealth.features import merge_clinical_features, feature_summary
from longihealth.split import split_patient_level
from longihealth.preprocess import (
    get_feature_columns,
    filter_features_train_only,
)


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def synthetic_cohort():
    """
    A small synthetic cohort with enough patients to allow
    stratified 70/15/15 splits (needs >= 20 patients).
    """

    n_patients = 20
    n_admissions = 3
    n = n_patients * n_admissions

    rng = np.random.default_rng(0)

    subjects = np.repeat(np.arange(1, n_patients + 1), n_admissions)
    hadm_ids = np.arange(1000, 1000 + n)
    stay_ids = np.arange(2000, 2000 + n)

    intimes = pd.date_range("2100-01-01", periods=n, freq="12h")

    # Ensure every patient has at least one positive outcome
    # to make stratification stable (each patient gets one 1).
    outcomes = np.zeros(n, dtype=int)
    outcomes[::3] = 1   # every third admission is positive

    df = pd.DataFrame({
        "subject_id": subjects,
        "hadm_id": hadm_ids,
        "stay_id": stay_ids,
        "intime": intimes,
        "outtime": intimes + pd.Timedelta(hours=48),
        "first_careunit": ["MICU"] * n,
        "last_careunit": ["MICU"] * n,
        "los": rng.uniform(1, 10, n),
        "gender": rng.choice(["M", "F"], n),
        "anchor_age": rng.integers(30, 90, n),
        "hospital_expire_flag": outcomes,
    })

    df["index_date"] = df["intime"]
    df["feature_window_start"] = df["index_date"]
    df["feature_window_end"] = df["index_date"] + pd.Timedelta(hours=24)
    df["outcome"] = df["hospital_expire_flag"].astype(int)

    return df


# ------------------------------------------------------------
# Config tests
# ------------------------------------------------------------

def test_config_loads(config):
    assert config["project"]["name"] == "LongiHealth"
    assert config["cohort"]["feature_window_hours"] == 24
    assert config["split"]["level"] == "patient"


def test_config_has_three_models(config):
    assert set(config["models"].keys()) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }


# ------------------------------------------------------------
# Cohort tests
# ------------------------------------------------------------

def test_cohort_summary_keys(synthetic_cohort):
    s = cohort_summary(synthetic_cohort)
    assert set(s.keys()) >= {
        "admissions",
        "patients",
        "positive_outcomes",
        "mortality_rate",
    }


def test_cohort_index_is_icu_intime(synthetic_cohort):
    assert (
        synthetic_cohort["index_date"]
        == synthetic_cohort["intime"]
    ).all()


def test_cohort_window_is_24h(synthetic_cohort):
    delta = (
        synthetic_cohort["feature_window_end"]
        - synthetic_cohort["feature_window_start"]
    )
    assert (delta == pd.Timedelta(hours=24)).all()


# ------------------------------------------------------------
# Feature-matrix tests
# ------------------------------------------------------------

def test_merge_clinical_features_no_leakage(synthetic_cohort):
    lab = synthetic_cohort[
        ["subject_id", "hadm_id", "outcome"]
    ].copy()
    lab["lab_mean_creatinine_50912"] = 1.0

    vital = synthetic_cohort[
        ["subject_id", "hadm_id"]
    ].copy()
    vital["vital_mean_heart_rate_220045"] = 80.0

    final = merge_clinical_features(
        synthetic_cohort,
        lab,
        vital,
    )

    for forbidden in [
        "hospital_expire_flag",
        "los",
        "deathtime",
        "dischtime",
    ]:
        assert forbidden not in final.columns


def test_feature_summary_counts(synthetic_cohort):
    lab = synthetic_cohort[
        ["subject_id", "hadm_id", "outcome"]
    ].copy()
    lab["lab_mean_creatinine_50912"] = 1.0
    lab["lab_min_creatinine_50912"] = 0.9

    vital = synthetic_cohort[
        ["subject_id", "hadm_id"]
    ].copy()
    vital["vital_mean_heart_rate_220045"] = 80.0

    final = merge_clinical_features(
        synthetic_cohort,
        lab,
        vital,
    )

    s = feature_summary(final)
    assert s["lab_features"] == 2
    assert s["vital_features"] == 1
    assert s["total_features"] == 3


# ------------------------------------------------------------
# Split tests
# ------------------------------------------------------------

def test_patient_level_split_no_overlap(synthetic_cohort, config):
    train, val, test = split_patient_level(synthetic_cohort, config)

    train_ids = set(train["subject_id"])
    val_ids = set(val["subject_id"])
    test_ids = set(test["subject_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_preserves_all_patients(synthetic_cohort, config):
    train, val, test = split_patient_level(synthetic_cohort, config)

    total = (
        train["subject_id"].nunique()
        + val["subject_id"].nunique()
        + test["subject_id"].nunique()
    )
    assert total == synthetic_cohort["subject_id"].nunique()


# ------------------------------------------------------------
# Preprocessing tests
# ------------------------------------------------------------

def test_train_only_filter_is_deterministic():
    """
    Build a small DataFrame with known-good and known-bad
    features, and verify the training-only filter behaves
    as expected.
    """

    n = 20
    rng = np.random.default_rng(1)

    df = pd.DataFrame({
        "subject_id": np.arange(1, n + 1),
        "hadm_id": np.arange(1000, 1000 + n),
        "stay_id": np.arange(2000, 2000 + n),
        "outcome": rng.integers(0, 2, n),
        # constant
        "lab_constant": 1.0,
        # all-NaN
        "lab_all_nan": np.nan,
        # varying (valid feature)
        "lab_varying": rng.normal(0, 1, n),
    })

    features = get_feature_columns(df)

    selected, removed = filter_features_train_only(
        df, features, missing_threshold=0.80
    )

    assert "lab_constant" in removed
    assert "lab_all_nan" in removed
    assert "lab_varying" in selected


def test_no_missing_after_simple_imputation():
    from sklearn.impute import SimpleImputer

    n = 20
    values = [1.0] * 10 + [np.nan] * 10

    df = pd.DataFrame({"lab_a": values})

    imputer = SimpleImputer(strategy="median")
    imputer.fit(df[["lab_a"]])

    out = imputer.transform(df[["lab_a"]])
    assert not np.isnan(out).any()
