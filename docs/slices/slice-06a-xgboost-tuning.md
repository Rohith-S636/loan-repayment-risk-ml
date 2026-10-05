# Slice 06A — Controlled XGBoost Tuning

## Module
`train` / `evaluate`

## What to build
Run a controlled, deterministic XGBoost hyperparameter experiment using the existing Slice 04 train/validation split.

The experiment is deliberately limited rather than exhaustive. XGBoost is the only model tuned because it was the strongest candidate in Slice 05.

## Selection protocol
- Use the existing training and validation applicant IDs from `models/preprocessing_bundle.joblib`.
- Reuse the existing leakage-safe preprocessor without refitting on validation or test data.
- Sample 20 XGBoost configurations with `random_state=42`.
- Optimize validation PR-AUC as the primary metric.
- Use validation ROC-AUC and F1 as tie-breakers.
- Test data is never loaded or inspected.
- Preserve the original `models/xgboost.joblib` until the tuned model is accepted.

## Tuned parameters
- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `reg_lambda`
- `reg_alpha`
- `scale_pos_weight`

The class-weight search varies the baseline negative/positive ratio by 0.75x, 1.00x, and 1.25x.

## Acceptance criteria
- [ ] All planned candidate trials complete successfully.
- [ ] Validation PR-AUC is recorded for every trial.
- [ ] Validation ROC-AUC, precision, recall, and F1 are recorded.
- [ ] Best configuration is selected using validation data only.
- [ ] Tuned model is saved separately as `models/xgboost_tuned.joblib`.
- [ ] Tuning metadata and trial results are saved.
- [ ] Test data is not used for tuning.

## Decision rule
Compare the tuned model against the Slice 05 baseline:
- Baseline validation PR-AUC: **0.259062**
- Baseline validation ROC-AUC: **0.772189**

Accept the tuned model only if it provides a meaningful validation improvement. If the improvement is negligible, retain the original XGBoost model.

## Required follow-up if accepted
1. Re-run Slice 07 threshold analysis using the accepted model.
2. Freeze the new validation-selected threshold.
3. Re-run Slice 08 final test evaluation exactly once.
4. Do not tune again after inspecting the new test results.
5. Use the accepted model and frozen threshold in Slice 09.
