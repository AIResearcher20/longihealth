"""
LongiHealth — Baseline classifiers.

Three models:
    1. Logistic Regression
    2. Random Forest
    3. Gradient Boosting

Each model is trained on the training split only. Class imbalance
is handled via class_weight / sample_weight.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.utils.class_weight import compute_sample_weight


# ------------------------------------------------------------
# Model builders
# ------------------------------------------------------------

def build_logistic_regression(
    config: Dict[str, Any]
) -> LogisticRegression:
    """
    Build a class-balanced logistic regression.
    """

    params = config["models"]["logistic_regression"]

    return LogisticRegression(
        C=params["C"],
        solver=params["solver"],
        max_iter=params["max_iter"],
        class_weight=params["class_weight"],
        random_state=config["project"]["random_seed"],
    )


def build_random_forest(
    config: Dict[str, Any]
) -> RandomForestClassifier:
    """
    Build a class-balanced random forest.
    """

    params = config["models"]["random_forest"]

    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight=params["class_weight"],
        random_state=config["project"]["random_seed"],
        n_jobs=-1,
    )


def build_gradient_boosting(
    config: Dict[str, Any]
) -> GradientBoostingClassifier:
    """
    Build a gradient boosting classifier.
    """

    params = config["models"]["gradient_boosting"]

    return GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        subsample=params["subsample"],
        random_state=config["project"]["random_seed"],
    )


# ------------------------------------------------------------
# Training helper
# ------------------------------------------------------------

def fit_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """
    Fit a model. For GradientBoosting, sample weights are used
    to handle class imbalance. Other models use their built-in
    class_weight parameter.
    """

    if isinstance(model, GradientBoostingClassifier):
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    return model


# ------------------------------------------------------------
# Feature importance helpers
# ------------------------------------------------------------

def feature_importance(
    model,
    feature_names: list[str],
) -> pd.DataFrame | None:
    """
    Return feature importance if available.
    """

    if hasattr(model, "feature_importances_"):
        return (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    if isinstance(model, LogisticRegression):
        coefs = np.abs(model.coef_).ravel()
        return (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": coefs,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    return None
