# Slice 01 — Data Validation

## Module
`data_loader`

## What to build
Validate all seven local CSVs, row/column counts, key columns, and table relationships.

## Acceptance criteria
- [ ] All seven expected files exist under `data/raw/`.
- [ ] Local counts match the inspection summary.
- [ ] `application_train` contains `SK_ID_CURR` and `TARGET`.
- [ ] Historical tables expose their expected keys.
- [ ] No raw CSV is tracked.

## Out of scope
Feature engineering and model training.
