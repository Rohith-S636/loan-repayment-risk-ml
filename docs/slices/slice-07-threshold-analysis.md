# Slice 07 — Threshold Analysis

## Module
`evaluate`

## What to build
Evaluate candidate probability thresholds on validation predictions and select one according to the documented precision/recall objective for the accepted final model.

## Acceptance criteria
- [x] Threshold results use validation data only.
- [x] Precision, recall, and F1 are shown across thresholds.
- [x] Selected threshold is recorded.
- [x] Test data is not used for selection.
- [x] The selected model is the final accepted baseline or tuned XGBoost model.

## Recorded result
The accepted tuned XGBoost model uses threshold **0.58**, selected by maximizing validation F1. At this threshold, validation precision is **0.251090**, recall is **0.475730**, and F1 is **0.328695**.

## Out of scope
Test-set threshold tuning and any model tuning after test inspection.
