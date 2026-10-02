# Architecture — Loan Repayment Risk Prediction

## Stack
- Python
- pandas / NumPy
- scikit-learn / XGBoost
- Matplotlib / Seaborn
- Streamlit
- joblib
- Local Home Credit CSV files

## System flow
```mermaid
flowchart LR
 A[Seven raw CSV tables] --> B[Validation]
 B --> C[Applicant-level aggregation]
 C --> D[Feature engineering]
 D --> E[Train / validation / test]
 E --> F[Leakage-safe preprocessing]
 F --> G[Model comparison]
 G --> H[Threshold analysis]
 H --> I[Final evaluation]
 I --> J[Explainability + demo]
```

## Modules
- `data_loader` — locate/read local source tables.
- `aggregation` — reduce historical tables to `SK_ID_CURR` aggregates; chunk large files.
- `feature_engineering` — financial ratios and compact repayment features.
- `preprocessing` — imputation, encoding, and scaling where required.
- `train` — train candidate models and persist the selected pipeline.
- `evaluate` — metrics, curves, confusion matrix, threshold analysis.
- `app` — lightweight Streamlit demo.

## Relationships
```text
application_train
  └── SK_ID_CURR
       ├── bureau
       │    └── bureau_balance via SK_ID_BUREAU
       └── previous_application
            ├── POS_CASH_balance via SK_ID_PREV
            ├── installments_payments via SK_ID_PREV
            └── credit_card_balance via SK_ID_PREV
```

The final modeling table has one row per `SK_ID_CURR`.

## Feature groups
- Application: selected current variables and valid ratios such as credit/income and annuity/income.
- Bureau: account, active/overdue, credit/debt/overdue, and compact bureau-balance history aggregates.
- Previous applications: count, approval/refusal, historical credit/application amounts.
- POS/CASH: history and DPD statistics.
- Installments: payment/installment ratios, late counts/ratios, payment delays.
- Credit card: history, balance/limit, and delinquency statistics.

Exact formulas must follow the actual CSV schema.

## Models
1. Logistic Regression — baseline.
2. Random Forest — nonlinear tree baseline.
3. XGBoost — boosting candidate.
4. HistGradientBoostingClassifier — fallback if XGBoost is unavailable/impractical.

Class weighting is the default imbalance strategy.

## Evaluation
ROC-AUC, PR-AUC, precision, recall, F1, and confusion matrix. Threshold selection uses validation data only; final test evaluation happens after the threshold is frozen.

## Constraints
- Raw data outside Git.
- Avoid loading all large tables simultaneously.
- No target leakage.
- Fixed random seed.
- Two-day deadline is a hard scope constraint.
