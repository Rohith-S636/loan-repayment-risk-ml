# Slice 05 — Model Training

## Module
`train`

## What to build
Train Logistic Regression, Random Forest, and XGBoost. Use HistGradientBoosting if XGBoost is unavailable or impractical.

## Acceptance criteria
- [x] Each attempted model trains successfully.
- [x] Class imbalance handling is enabled where supported.
- [x] Probability predictions are available.
- [x] Random seed/settings are recorded.
- [x] Model comparison results are saved.
- [x] Baseline model comparison is complete before any optional tuning.

## Recorded result
XGBoost was the strongest baseline, with validation ROC-AUC **0.772189** and PR-AUC **0.259062**, and was therefore selected as the only model for controlled tuning in Slice 06A.

## Out of scope
Exhaustive hyperparameter tuning. Controlled XGBoost tuning is handled separately in Slice 06A.
