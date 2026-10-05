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
- Logistic Regression, Random Forest, and XGBoost.
- Controlled non-exhaustive XGBoost tuning.
- Class-imbalance handling.
- ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix.
- Validation-only model selection and threshold analysis.
- Permutation feature importance.
- Lightweight Streamlit demo.

## Model-selection requirements
- XGBoost is the only tuned model because it was the strongest baseline candidate.
- Tuning uses the fixed Slice 04 train/validation split.
- Validation PR-AUC is the primary tuning metric.
- Test data is completely excluded from tuning and model selection.
- If tuning is accepted, threshold selection and final test evaluation are repeated.
- No tuning is permitted after final test inspection.

## Accepted experiment result
- Accepted model: tuned XGBoost (`models/xgboost_tuned.joblib`).
- Validation PR-AUC improved from **0.259062** to **0.261614**.
- Validation ROC-AUC improved from **0.772189** to **0.774488**.
- Validation-selected threshold: **0.58**, chosen by maximizing F1.
- Final untouched-test ROC-AUC: **0.778114**.
- Final untouched-test PR-AUC: **0.273767**.
- Final untouched-test precision: **0.256052**.
- Final untouched-test recall: **0.472910**.
- Final untouched-test F1: **0.332225**.

## Out of scope
- Exhaustive hyperparameter optimization across all models.
- K-means/PCA/t-SNE.
- Deep learning.
- Polynomial expansion.
- Cloud deployment.
- Complex production frontend.
- Automatic dataset downloading.

## Success criteria
- All seven local datasets can be processed without committing raw data.
- One applicant-level modeling table is produced.
- Candidate models train and produce probabilities.
- A controlled XGBoost tuning experiment is reproducible.
- The selected model is determined from validation data only.
- Threshold is selected on validation data and then frozen.
- Final test results, feature importance, and demo are available.
- README, two-page report, and presentation use the recorded results above.
