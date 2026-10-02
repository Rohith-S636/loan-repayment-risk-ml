# PRD — Loan Repayment Risk Prediction

## Goal
Build a working ML system that predicts loan repayment difficulty at applicant level using current application data plus aggregated historical credit and repayment behavior.

## Users
- ML mini-project evaluators/faculty.
- Project team members running the reproducible pipeline.
- Demo users inspecting a risk probability and classification.

## Core features
- Seven-table Home Credit ingestion.
- Applicant-level historical aggregation.
- Compact financial and repayment feature engineering.
- Leakage-safe preprocessing.
- Logistic Regression, Random Forest, and XGBoost/fallback.
- Class-imbalance handling.
- ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix.
- Validation-based threshold analysis.
- Permutation feature importance.
- Lightweight Streamlit demo.

## Out of scope
- K-means/PCA/t-SNE.
- Deep learning.
- Polynomial expansion.
- Exhaustive hyperparameter optimization.
- Cloud deployment.
- Complex production frontend.
- Automatic dataset downloading.

## Success criteria
- All seven local datasets can be processed without committing raw data.
- One applicant-level modeling table is produced.
- Candidate models train and produce probabilities.
- Core metrics are recorded.
- Threshold is selected on validation data and then frozen.
- Final test results, feature importance, and demo are available.
- README, two-page report, and presentation can be produced from recorded results.
