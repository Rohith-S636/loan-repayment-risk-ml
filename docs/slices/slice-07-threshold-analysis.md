# Slice 07 — Threshold Analysis

## Module
`evaluate`

## What to build
Evaluate candidate probability thresholds on validation predictions and select one according to the documented precision/recall objective for the accepted final model.

## Acceptance criteria
- [ ] Threshold results use validation data only.
- [ ] Precision, recall, and F1 are shown across thresholds.
- [ ] Selected threshold is recorded.
- [ ] Test data is not used for selection.
- [ ] The selected model is the final accepted baseline or tuned XGBoost model.

## Out of scope
Test-set threshold tuning and any model tuning after test inspection.
