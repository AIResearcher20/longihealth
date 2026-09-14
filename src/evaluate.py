"""
LongiHealth — Evaluation utilities.

Computes standard classification metrics on validation and
test splits, saves structured JSON reports, and provides
comparison tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
)


METRIC_NAMES = [
    "AUROC",
    "AUPRC",
    "F1",
    "Precision",
    "Recall",
    "Balanced_Accuracy",
    "Brier",
]


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    """

    predictions = (probabilities >= threshold).astype(int)

    return {
        "AUROC": float(roc_auc_score(y_true, probabilities)),
        "AUPRC": float(average_precision_score(y_true, probabilities)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "Brier": float(brier_score_loss(y_true, probabilities)),
    }


def confusion(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Return confusion matrix.
    """

    preds = (probabilities >= threshold).astype(int)
    return confusion_matrix(y_true, preds)


def compare_models(
    model_predictions: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Build a comparison table across models and splits.

    Parameters
    ----------
    model_predictions :
        {
            "ModelName": {
                "validation": {"y_true": ..., "proba": ...},
                "test":       {"y_true": ..., "proba": ...},
            },
            ...
        }
    """

    rows = []

    for model_name, splits in model_predictions.items():
        for split_name, payload in splits.items():
            metrics = evaluate_predictions(
                payload["y_true"],
                payload["proba"],
                threshold=threshold,
            )
            rows.append({
                "model": model_name,
                "split": split_name,
                **metrics,
            })

    return pd.DataFrame(rows)


def build_report(
    cohort_summary: Dict[str, Any],
    split_summary: Dict[str, Any],
    feature_summary: Dict[str, Any],
    comparison: pd.DataFrame,
    best_test_model: str,
    limitations: list[str],
) -> Dict[str, Any]:
    """
    Build the structured evaluation report.
    """

    return {
        "project": "LongiHealth",
        "task": "Exploratory hospital mortality prediction",
        "dataset": "MIMIC-IV Clinical Database Demo",
        "cohort": cohort_summary,
        "splits": split_summary,
        "features": feature_summary,
        "models": comparison.to_dict(orient="records"),
        "best_test_model_by_AUROC": best_test_model,
        "limitations": limitations,
    }


def save_report(
    report: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save the report as JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return output_path


def best_model_by_metric(
    comparison: pd.DataFrame,
    metric: str = "AUROC",
    split: str = "test",
) -> str:
    """
    Return the name of the best model on a given split.
    """

    subset = comparison[comparison["split"] == split]

    if subset.empty:
        raise ValueError(f"No rows for split '{split}'.")

    idx = subset[metric].idxmax()
    return subset.loc[idx, "model"]
