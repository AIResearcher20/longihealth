from typing import Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.utils.class_weight import compute_sample_weight


def build_logistic_regression(
    config: dict[str, Any]
) -> LogisticRegression:
    params = config["models"]["logistic_regression"]

    return LogisticRegression(
        C=params["C"],
        solver=params["solver"],
        max_iter=params["max_iter"],
        class_weight=params["class_weight"],
        random_state=config["project"]["random_seed"],
    )


def build_random_forest(
    config: dict[str, Any]
) -> RandomForestClassifier:
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
    config: dict[str, Any]
) -> GradientBoostingClassifier:
    params = config["models"]["gradient_boosting"]

    return GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        subsample=params["subsample"],
        random_state=config["project"]["random_seed"],
    )


def fit_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    if isinstance(model, GradientBoostingClassifier):
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    return model


def feature_importance(
    model,
    feature_names: list[str],
) -> pd.DataFrame | None:
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
