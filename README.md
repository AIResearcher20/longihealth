LongiHealth

Reproducible Longitudinal EHR Analytics Pipeline

LongiHealth is a reproducible research software project for longitudinal electronic health record (EHR) analytics and exploratory outcome prediction using the MIMIC-IV Clinical Database Demo v2.2.

The project was designed around a central research-software question:

«Can a reproducible, leakage-aware clinical data pipeline transform heterogeneous early-admission EHR measurements into a structured feature space suitable for exploratory outcome modeling?»

Rather than focusing only on predictive accuracy, LongiHealth emphasizes data engineering, temporal feature construction, leakage prevention, reproducibility, automated evaluation, and transparent interpretation of model limitations.

---

Research Questions

RQ1 — Cohort construction

Can an admission-level ICU cohort be constructed reproducibly from heterogeneous hospital and ICU tables while preserving a well-defined temporal index?

RQ2 — Temporal clinical representation

Can laboratory measurements and vital signs observed during the first 24 hours of an ICU stay be transformed into a structured, admission-level feature representation?

RQ3 — Leakage prevention

Can patient-level splitting and training-only preprocessing prevent information leakage across repeated admissions and between training and evaluation sets?

RQ4 — Exploratory outcome modeling

How do simple baseline machine-learning models compare when predicting in-hospital mortality from the resulting clinical feature representation?

RQ5 — Reproducibility

Can the complete analytical workflow be packaged using software-engineering practices such as configuration management, testing, Docker, dependency specification, and continuous integration?

---

Study Design

LongiHealth uses an admission-level cohort.

For each selected admission:

Hospital admission
        │
        ▼
First ICU stay
        │
        ▼
Index date = ICU admission time
        │
        ├── First 24 hours
        │      ├── Laboratory measurements
        │      └── Vital signs
        │
        ▼
Clinical feature representation
        │
        ▼
Patient-level train / validation / test split
        │
        ▼
Leakage-safe preprocessing
        │
        ▼
Baseline machine-learning models
        │
        ▼
Automated evaluation

The outcome is hospital mortality, represented by the admission-level "hospital_expire_flag".

Importantly, the outcome and post-outcome variables were excluded from the predictive feature space.

---

Dataset

The project uses the MIMIC-IV Clinical Database Demo v2.2.

The demonstration cohort contains:

- 100 unique patients
- 128 unique hospital admissions
- 128 first ICU stays associated with the selected admissions
- 15 positive mortality outcomes
- Mortality rate: 11.72%

Because this is the MIMIC-IV demonstration dataset, the cohort is intentionally small and is used for methodological and software-engineering demonstration rather than clinical validation.

Raw MIMIC-IV data are not included in this repository.

---

Feature Engineering

Clinical information was extracted from two major sources.

Laboratory measurements

Laboratory events were restricted to the first 24 hours after the ICU index time.

For each laboratory item, the pipeline calculated:

- measurement count
- mean
- minimum
- maximum
- standard deviation
- abnormal measurement count

The laboratory extraction initially produced:

- 192 laboratory items
- 1,152 item/statistic combinations
- 336 selected laboratory features after leakage-safe training-set filtering

Vital signs

The pipeline extracted clinically relevant ICU measurements including:

- heart rate
- arterial blood pressure
- non-invasive blood pressure
- mean arterial pressure
- temperature
- oxygen saturation

For each vital-sign item, the pipeline calculated:

- count
- mean
- minimum
- maximum
- standard deviation

This produced:

- 50 vital-sign features

---

Final Clinical Representation

The final clinical representation contained:

Feature group| Features
Laboratory measurements| 336
Vital signs| 50
Total| 386

The final admission-level matrix contained:

128 admissions × 386 clinical features

with an overall feature missingness of approximately 40.62% before final preprocessing.

---

Leakage Prevention

Leakage prevention was treated as a central methodological requirement.

Patient-level splitting

Patients were divided into:

- Training: 70 patients
- Validation: 15 patients
- Test: 15 patients

Corresponding admissions:

- Training: 89
- Validation: 20
- Test: 19

There was no patient overlap between the three partitions.

This is important because individual patients can contribute multiple hospital admissions. Random admission-level splitting could otherwise allow information from the same patient to appear in both training and evaluation sets.

Training-only preprocessing

Feature selection was performed using the training set.

The following preprocessing operations were fitted exclusively on training data:

1. Missingness-based feature filtering
2. Constant-feature removal
3. Median imputation
4. Standardization

The resulting transformations were then applied unchanged to validation and test data.

This design prevents evaluation data from influencing preprocessing decisions.

---

Machine-Learning Baselines

Three baseline classifiers were evaluated:

Logistic Regression

A linear baseline with class weighting was used to establish a simple reference model.

Random Forest

A tree-based ensemble was used to capture nonlinear relationships and feature interactions.

Gradient Boosting

A gradient-boosted decision-tree model was used as a second nonlinear baseline.

---

Evaluation

The pipeline evaluates models using multiple complementary metrics:

- AUROC
- AUPRC
- F1 score
- Balanced accuracy
- Brier score
- Confusion matrix

Because the demonstration cohort contains only 15 positive outcomes, the evaluation deliberately avoids treating any single metric as definitive.

---

Results

Model comparison

Model| Split| AUROC| AUPRC| F1| Balanced Accuracy| Brier
Logistic Regression| Validation| 0.5833| 0.1623| 0.0000| 0.4167| 0.1940
Logistic Regression| Test| 0.2353| 0.1026| 0.0000| 0.2647| 0.4355
Random Forest| Test| 0.7059| 0.2250| 0.0000| 0.5000| 0.1137
Gradient Boosting| Test| 0.6765| 0.2083| 0.0000| 0.3824| 0.1792

The Random Forest produced the strongest exploratory test AUROC:

AUROC = 0.7059

with:

AUPRC = 0.2250

and:

Brier score = 0.1137

However, these results should not be interpreted as evidence of clinical predictive performance.

---

Interpretation

The results demonstrate several methodological observations.

First, the nonlinear tree-based models performed better than the linear baseline on the small demonstration test set. Random Forest achieved the highest exploratory AUROC, while Gradient Boosting produced a similar but slightly lower ranking.

Second, the classification metrics reveal an important limitation: the models did not achieve positive-class F1 performance on the reported test split. This highlights the difficulty of evaluating mortality prediction with only two positive test cases.

Third, the relatively strong AUROC of the Random Forest should therefore be interpreted cautiously. With only two positive outcomes in the test set, a small change in prediction ranking can substantially alter the reported metric.

The purpose of this project is consequently not to claim a clinically useful mortality predictor, but to demonstrate a reproducible analytical framework capable of handling longitudinal clinical data while explicitly exposing its statistical limitations.

---

Feature Importance

The Random Forest analysis identified several laboratory-derived variables among the highest-ranked features.

Examples include:

- creatinine abnormal-count
- anion-gap minimum
- creatinine minimum
- anion-gap mean
- creatinine mean
- MCHC minimum
- PTT standard deviation
- urea nitrogen mean
- creatinine maximum

These findings are treated as model-specific associations rather than causal or clinical conclusions.

Feature importance does not establish that any individual biomarker causes mortality.

---

Research Software Engineering

LongiHealth was designed as a research software project rather than a one-off analysis notebook.

The workflow incorporates:

- modular data-processing stages
- explicit temporal windows
- leakage audits
- patient-level splitting
- training-only preprocessing
- structured configuration
- reproducible model settings
- automated evaluation
- JSON-based reporting
- dependency management
- Docker
- pytest-based validation
- GitHub Actions CI

The intended execution pattern is:

PYTHONPATH=src python -m longihealth.pipeline

The Docker environment provides a reproducible execution target:

docker build -t longihealth .
docker run --rm longihealth

---

Reproducibility

The project separates:

Source code

Reusable analytical and pipeline components.

Configuration

Model and feature-engineering parameters.

Reports

Aggregate evaluation results and structured JSON summaries.

Figures

Model-comparison and feature-importance visualizations.

Clinical data

Raw and patient-level derived clinical data are intentionally excluded from version control.

---

Limitations

Several limitations are fundamental to this study.

1. Small demonstration cohort
   The MIMIC-IV Demo contains only 100 patients.

2. Very small test set
   Only 19 admissions are included in the test partition.

3. Very few positive outcomes
   The test set contains only two positive outcomes.

4. Metric instability
   AUROC, AUPRC, F1, and related metrics can change substantially with such a small evaluation sample.

5. No clinical validation
   The models are exploratory and have not been externally validated.

6. No causal interpretation
   Feature importance represents model behavior and does not establish biological or clinical causality.

7. Demonstration dataset
   Results cannot be generalized to the full MIMIC-IV population or to real-world clinical deployment.

---

Project Contribution

The main contribution of LongiHealth is therefore not a claim of superior clinical prediction.

Instead, the project demonstrates a complete reproducible workflow for:

«heterogeneous EHR data → temporal cohort construction → clinical feature engineering → leakage-safe preprocessing → patient-level evaluation → reproducible machine learning → automated reporting»

This makes LongiHealth particularly relevant to research software engineering, biomedical data science, clinical machine learning, and population-scale health-data analytics.

---

Repository Structure

LongiHealth/
│
├── LongiHealth_Professional_Notebook.ipynb
├── README.md
├── Dockerfile
├── requirements.txt
├── .gitignore
├── PACKAGE_CONTENTS.md
│
├── src/
│   └── longihealth/
│       ├── ingestion/
│       ├── validation/
│       ├── cohort/
│       ├── preprocessing/
│       ├── features/
│       ├── models/
│       ├── evaluation/
│       └── reporting/
│
├── tests/
│
├── figures/
│   ├── test_auroc_comparison.png
│   ├── test_auprc_comparison.png
│   ├── test_brier_comparison.png
│   └── random_forest_top15_features.png
│
├── reports/
│   ├── model_comparison.csv
│   └── longihealth_evaluation_report.json
│
└── .github/
    └── workflows/
        └── ci.yml

---

Status

Project status: Completed exploratory research prototype

The analytical workflow, leakage controls, baseline models, evaluation, reporting, containerization, and CI configuration have been developed.

The project is intended as a research software portfolio project demonstrating the ability to design reproducible computational workflows for biomedical and longitudinal health data.
