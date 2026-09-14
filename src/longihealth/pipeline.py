"""
LongiHealth — End-to-end pipeline.

Run:
    python -m longihealth.pipeline

Executes:
    1. Load configuration
    2. Build cohort
    3. Extract lab features
    4. Extract vital features
    5. Merge final clinical matrix
    6. Patient-level split
    7. Leakage-safe preprocessing
    8. Train baseline models
    9. Evaluate and save reports
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_config
from .cohort import build_cohort, cohort_summary
from .data import load_dictionary
from .features_lab import extract_lab_features, build_lab_matrix
from .features_vital import extract_vital_features, build_vital_matrix
from .features import merge_clinical_features, feature_summary
from .split import split_patient_level, split_summary
from .preprocess import preprocess_splits, save_preprocessed
from .models import (
    build_logistic_regression,
    build_random_forest,
    build_gradient_boosting,
    fit_model,
)
from .evaluate import (
    compare_models,
    build_report,
    save_report,
    best_model_by_metric,
)


def run_pipeline(config_path: str | Path | None = None) -> None:
    """
    Run the LongiHealth pipeline end to end.
    """

    print("=" * 70)
    print("LongiHealth — Pipeline")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Config
    # --------------------------------------------------------

    cfg = load_config(config_path)
    artifacts = Path(cfg["paths"]["artifacts_root"])
    artifacts.mkdir(parents=True, exist_ok=True)

    print(f"\n[1] Config loaded: {cfg['project']['name']} v{cfg['project']['version']}")

    # --------------------------------------------------------
    # 2. Cohort
    # --------------------------------------------------------

    cohort = build_cohort(cfg)
    cohort.to_csv(artifacts / "longihealth_cohort.csv", index=False)

    cs = cohort_summary(cohort)
    print(f"[2] Cohort: {cs['admissions']} admissions, {cs['patients']} patients, "
          f"{cs['positive_outcomes']} positive outcomes")

    # --------------------------------------------------------
    # 3. Lab features
    # --------------------------------------------------------

    lab_features = extract_lab_features(cfg, cohort)
    d_labitems = load_dictionary(cfg, "d_labitems")

    lab_matrix = build_lab_matrix(
        lab_features,
        cohort,
        min_observed_fraction=cfg["features"]["lab"]["min_observed_fraction"],
        dictionary=d_labitems,
    )
    lab_matrix.to_csv(artifacts / "longihealth_lab_matrix_qc.csv", index=False)

    lab_feat_cols = [
        c for c in lab_matrix.columns
        if c.startswith(("lab_", "abnormal_count_"))
    ]
    print(f"[3] Lab features: {len(lab_feat_cols)}")

    # --------------------------------------------------------
    # 4. Vital features
    # --------------------------------------------------------

    vital_features = extract_vital_features(cfg, cohort)
    d_items = load_dictionary(cfg, "d_items")

    vital_matrix = build_vital_matrix(
        vital_features,
        cohort,
        d_items,
    )
    vital_matrix.to_csv(artifacts / "longihealth_vital_matrix.csv", index=False)

    vital_feat_cols = [
        c for c in vital_matrix.columns
        if c.startswith("vital_")
    ]
    print(f"[4] Vital features: {len(vital_feat_cols)}")

    # --------------------------------------------------------
    # 5. Final clinical matrix
    # --------------------------------------------------------

    clinical = merge_clinical_features(cohort, lab_matrix, vital_matrix)
    clinical.to_csv(artifacts / "longihealth_final_feature_matrix.csv", index=False)

    fs = feature_summary(clinical)
    print(f"[5] Final matrix: {clinical.shape}, "
          f"{fs['total_features']} features, "
          f"missing {fs['overall_missingness']:.2%}")

    # --------------------------------------------------------
    # 6. Split
    # --------------------------------------------------------

    train, val, test = split_patient_level(clinical, cfg)

    train.to_csv(artifacts / "longihealth_train.csv", index=False)
    val.to_csv(artifacts / "longihealth_validation.csv", index=False)
    test.to_csv(artifacts / "longihealth_test.csv", index=False)

    ss = split_summary(train, val, test)
    print(f"[6] Split: train={ss['train']['admissions']}, "
          f"val={ss['validation']['admissions']}, "
          f"test={ss['test']['admissions']}")

    # --------------------------------------------------------
    # 7. Preprocessing
    # --------------------------------------------------------

    prep = preprocess_splits(train, val, test, cfg)
    save_preprocessed(prep, cfg["paths"]["preprocessed_root"])

    print(f"[7] Preprocessing: {len(prep['features'])} features kept, "
          f"{len(prep['removed'])} removed")

    # --------------------------------------------------------
    # 8. Models
    # --------------------------------------------------------

    X_train = prep["X_train"]
    X_val = prep["X_val"]
    X_test = prep["X_test"]

    y_train = prep["y_train"]
    y_val = prep["y_val"]
    y_test = prep["y_test"]

    predictions = {}

    for name, builder in [
        ("Logistic Regression", build_logistic_regression),
        ("Random Forest", build_random_forest),
        ("Gradient Boosting", build_gradient_boosting),
    ]:
        model = builder(cfg)
        fit_model(model, X_train, y_train)

        predictions[name] = {
            "validation": {
                "y_true": y_val,
                "proba": model.predict_proba(X_val)[:, 1],
            },
            "test": {
                "y_true": y_test,
                "proba": model.predict_proba(X_test)[:, 1],
            },
        }

    print(f"[8] Trained {len(predictions)} models")

    # --------------------------------------------------------
    # 9. Evaluation
    # --------------------------------------------------------

    threshold = cfg["evaluation"]["threshold"]
    comparison = compare_models(predictions, threshold=threshold)

    eval_dir = Path(cfg["paths"]["evaluation_root"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(eval_dir / "model_comparison.csv", index=False)

    best = best_model_by_metric(comparison, "AUROC", "test")

    report = build_report(
        cohort_summary=cs,
        split_summary=ss,
        feature_summary=fs,
        comparison=comparison,
        best_test_model=best,
        limitations=[
            "MIMIC-IV Demo contains a very small cohort.",
            "The test set contains only 2 positive outcomes.",
            "Test metrics are exploratory and unstable.",
            "Results should not be interpreted as clinical performance.",
        ],
    )

    save_report(report, eval_dir / "longihealth_evaluation_report.json")

    print(f"[9] Best test model by AUROC: {best}")
    print(f"\n✅ Pipeline completed.")
    print(f"   Artifacts root: {artifacts}")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
