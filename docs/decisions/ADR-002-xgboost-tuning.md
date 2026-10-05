# ADR-002 — Controlled XGBoost Tuning

## Status
Proposed experiment — final acceptance depends on validation results.

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

## Acceptance rule
The tuned model is accepted only when its validation improvement over the baseline is meaningful. If the gain is negligible, retain the original XGBoost model.

## Consequence
If the tuned model is accepted:
1. Rerun Slice 07 threshold selection.
2. Freeze the new threshold.
3. Rerun Slice 08 final test evaluation once.
4. Use the accepted model and threshold in Slice 09.
5. Do not tune again after inspecting the new test results.

The previously generated Slice 08 test metrics remain useful as the baseline experiment record, but they are not the final model results if a tuned model is subsequently accepted.
