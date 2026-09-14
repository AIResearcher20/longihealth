"""
LongiHealth — Leakage-safe preprocessing.

Rules
-----
- Feature filtering, imputation, and scaling are ALL fit on the
  training split only.
- Validation and test splits are transformed using the fitted
  objects (never refit).
- No imputation or scaling is computed on validation/test data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


NON_FEATURE_COLUMNS: List[str] = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "index_date",
    "outcome",
]


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return columns that are model inputs.
    """

    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLUMNS
    ]


def filter_features_train_only(
    train: pd.DataFrame,
    feature_columns: List[str],
    missing_threshold: float,
) -> Tuple[List[str], List[str]]:
    """
    Identify features to keep, based strictly on training data.

    Returns
    -------
    (selected_features, removed_features)
    """

    train_features = train[feature_columns]

    missingness = train_features.isna().mean()
    nunique = train_features.nunique(dropna=True)

    too_missing = missingness[
        missingness > missing_threshold
    ].index.tolist()

    constant = nunique[nunique <= 1].index.tolist()

    removed = sorted(set(too_missing) | set(constant))

    selected = [
        c for c in feature_columns
        if c not in removed
    ]

    return selected, removed


def fit_preprocessing(
    X_train: pd.DataFrame,
) -> Tuple[SimpleImputer, StandardScaler]:
    """
    Fit imputer + scaler on training data only.
    """

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)

    X_imp = imputer.transform(X_train)

    scaler = StandardScaler()
    scaler.fit(X_imp)

    return imputer, scaler


def transform(
    X: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Apply fitted imputer + scaler to a split.
    """

    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    return X_scaled


def preprocess_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline for train / val / test.
    """

    missing_threshold = config["preprocessing"]["missing_threshold"]

    all_features = get_feature_columns(train)

    selected, removed = filter_features_train_only(
        train, all_features, missing_threshold
    )

    # --------------------------------------------------------
    # Build X / y
    # --------------------------------------------------------

    X_train = train[selected].copy()
    X_val = val[selected].copy()
    X_test = test[selected].copy()

    y_train = train["outcome"].astype(int).values
    y_val = val["outcome"].astype(int).values
    y_test = test["outcome"].astype(int).values

    # --------------------------------------------------------
    # Fit only on train
    # --------------------------------------------------------

    imputer, scaler = fit_preprocessing(X_train)

    X_train_s = transform(X_train, imputer, scaler)
    X_val_s = transform(X_val, imputer, scaler)
    X_test_s = transform(X_test, imputer, scaler)

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert X_train_s.shape[1] == len(selected)
    assert X_val_s.shape[1] == len(selected)
    assert X_test_s.shape[1] == len(selected)

    assert not np.isnan(X_train_s).any()
    assert not np.isnan(X_val_s).any()
    assert not np.isnan(X_test_s).any()

    return {
        "features": selected,
        "removed": removed,
        "X_train": X_train_s,
        "X_val": X_val_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "imputer": imputer,
        "scaler": scaler,
    }


def save_preprocessed(
    result: Dict[str, Any],
    output_dir: str | Path,
) -> None:
    """
    Save preprocessed matrices and feature list to disk.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = result["features"]

    df_train = pd.DataFrame(result["X_train"], columns=features)
    df_val = pd.DataFrame(result["X_val"], columns=features)
    df_test = pd.DataFrame(result["X_test"], columns=features)

    df_train["outcome"] = result["y_train"]
    df_val["outcome"] = result["y_val"]
    df_test["outcome"] = result["y_test"]

    df_train.to_csv(output_dir / "X_train.csv", index=False)
    df_val.to_csv(output_dir / "X_validation.csv", index=False)
    df_test.to_csv(output_dir / "X_test.csv", index=False)

    pd.Series(features, name="feature").to_csv(
        output_dir / "selected_features.csv", index=False
    )
