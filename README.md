# Loan Repayment Risk Prediction Using Machine Learning

**Working title:** Loan Repayment Risk Prediction Using Machine Learning: An Imbalance-Aware and Interpretable Approach

A two-day ML mini-project for predicting loan repayment difficulty using the Home Credit multi-table dataset.

## Objective
Build a reproducible applicant-level pipeline combining current application data with historical credit and repayment behavior, handling class imbalance, comparing ML models, optimizing the decision threshold on validation data, and producing an interpretable demo.

## In scope
- All seven Home Credit source tables.
- Applicant-level historical aggregation.
- Compact financial and repayment features.
- Stratified train/validation/test split.
- Leakage-safe preprocessing.
- Logistic Regression, Random Forest, XGBoost (HistGradientBoosting fallback).
- Class weighting as the default imbalance strategy.
- ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix.
- Validation-based threshold analysis.
- Permutation feature importance.
- Lightweight Streamlit demo.
- Two-page report and presentation.

## Out of scope
K-means, PCA/t-SNE, polynomial expansion, deep learning, exhaustive tuning, cloud deployment, complex frontend, and SMOTE unless time permits.

## Dataset
| Dataset | Rows | Columns |
|---|---:|---:|
| application_train.csv | 307,511 | 122 |
| bureau.csv | 1,716,428 | 17 |
| bureau_balance.csv | 27,299,925 | 3 |
| previous_application.csv | 1,670,214 | 37 |
| POS_CASH_balance.csv | 10,001,358 | 8 |
| installments_payments.csv | 13,605,401 | 8 |
| credit_card_balance.csv | 3,840,312 | 23 |

The raw CSVs are approximately 2.3 GB and are **not committed to GitHub**. Place them in `data/raw/`.

## Pipeline
```text
7 raw tables → validation → applicant-level aggregation
→ financial/repayment features → train/validation/test
→ leakage-safe preprocessing → 3 model comparison
→ imbalance evaluation → threshold analysis
→ final test evaluation → feature importance → demo
```

## Local setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducibility rules
- Fixed random seed.
- One modeling row per `SK_ID_CURR`.
- Never use `TARGET` to construct predictors.
- Fit learned preprocessing on training data only.
- Select the classification threshold on validation data only.
- Keep the test set untouched until final evaluation.
- Record actual experimental results; do not predeclare a winning model.

See `docs/` for the project specification, architecture, two-day sprints, and implementation slices.
