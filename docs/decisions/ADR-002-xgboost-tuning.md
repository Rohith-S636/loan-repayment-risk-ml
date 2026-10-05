# ADR-002 — Controlled XGBoost Tuning

## Status
Accepted and completed.

## Context
Slice 05 established XGBoost as the strongest baseline model. The baseline validation results were:

- ROC-AUC: **0.772189**
- PR-AUC: **0.259062**
- F1 at threshold 0.50: **0.291262**

A limited tuning experiment can test whether the selected boosting model can improve without changing the data split or introducing test-set leakage.

## Decision
Run a deterministic, non-exhaustive XGBoost tuning experiment.

### Search
Twenty configurations are sampled with random seed 42.

Parameters:
- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `reg_lambda`
- `reg_alpha`
- `scale_pos_weight`

### Selection
- Primary metric: validation PR-AUC.
- Tie-breaker 1: validation ROC-AUC.
- Tie-breaker 2: validation F1.
- The Slice 04 train/validation IDs are reused exactly.
- The existing preprocessing bundle is reused without refitting on validation or test data.
- The test split is never loaded by the tuning script.

### Artifacts
- `results/tuning/xgboost_trials.csv`
- `results/tuning/best_params.json`
- `results/tuning/tuning_metadata.json`
- `models/xgboost_tuned.joblib`

## Observed result and acceptance
The tuned candidate improved validation PR-AUC from **0.259062** to **0.261614** and validation ROC-AUC from **0.772189** to **0.774488**. The tuned model was accepted as the final model because it improved the primary validation metric under the predefined selection protocol.

## Completed consequence
1. Slice 07 threshold selection was rerun for the accepted model.
2. Threshold **0.58** was selected on validation data by maximizing F1.
3. Slice 08 final test evaluation was rerun once with the frozen model and threshold.
4. Slice 09 uses the accepted model and frozen threshold.
5. No tuning was performed after final test inspection.

The final accepted-model test results are ROC-AUC **0.778114**, PR-AUC **0.273767**, precision **0.256052**, recall **0.472910**, and F1 **0.332225**.
