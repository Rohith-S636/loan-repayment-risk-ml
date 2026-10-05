# Loan Repayment Risk Prediction Using Machine Learning

**Working title:** Loan Repayment Risk Prediction Using Machine Learning: An Imbalance-Aware and Interpretable Approach

A two-day ML mini-project for predicting loan repayment difficulty using the Home Credit multi-table dataset.

## Objective
Build a reproducible applicant-level pipeline combining current application data with historical credit and repayment behavior, handling class imbalance, comparing ML models, performing a controlled XGBoost tuning experiment, optimizing the decision threshold on validation data, and producing an interpretable demo.

## In scope
- All seven Home Credit source tables.
- Applicant-level historical aggregation.
- Compact financial and repayment features.
- Stratified train/validation/test split.
- Leakage-safe preprocessing.
- Logistic Regression, Random Forest, and XGBoost.
- Controlled, non-exhaustive XGBoost tuning.
- Class weighting as the default imbalance strategy.
- ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix.
- Validation-based model and threshold selection.
- Permutation feature importance.
- Lightweight Streamlit demo.
- Two-page report and presentation.

## Out of scope
Exhaustive hyperparameter search across all models, K-means, PCA/t-SNE, polynomial expansion, deep learning, complex frontend, cloud deployment, and SMOTE unless separately justified and time permits.

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
→ leakage-safe preprocessing → baseline model comparison
→ controlled XGBoost tuning → validation model selection
→ threshold analysis → frozen final evaluation
→ feature importance → Streamlit demo
```

## Model-selection protocol
The initial Slice 05 baseline selected XGBoost on validation performance:
- Baseline validation ROC-AUC: **0.772189**
- Baseline validation PR-AUC: **0.259062**
- Baseline validation F1 at 0.50: **0.291262**

A controlled XGBoost tuning experiment is implemented separately in Slice 06A. It uses the fixed Slice 04 train/validation split, selects by validation PR-AUC, and never uses the test set. The tuned model is accepted only if the validation improvement is meaningful. If accepted, threshold selection and final test evaluation are rerun with the tuned model.

## Reproducibility rules
- Fixed random seed: 42.
- One modeling row per `SK_ID_CURR`.
- Never use `TARGET` to construct predictors.
- Fit learned preprocessing on training data only.
- Tune/select models using training/validation data only.
- Select the classification threshold on validation data only.
- Keep the test set untouched until final evaluation.
- Never tune after inspecting final test results.
- Record actual experimental results; do not predeclare a winning tuned model.

## Local setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Controlled tuning
Run after the baseline pipeline is available:

```bash
python src/tune_xgboost.py
```

The experiment writes trial results under `results/tuning/` and saves the candidate model as `models/xgboost_tuned.joblib`. Do not replace the baseline model until the validation comparison is reviewed.

See `docs/` for the project specification, architecture, research decisions, sprints, and implementation slices.
