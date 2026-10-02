# Sprint 01 — Data and Working Models

## Goal
By the end of Day 1, the project has a reproducible applicant-level dataset and working candidate models with recorded metrics.

## Timebox
Day 1.

## Slices
- [ ] `slice-01-data-validation` — validate the seven local datasets and keys.
- [ ] `slice-02-historical-aggregation` — aggregate historical tables to applicant level.
- [ ] `slice-03-feature-engineering` — create compact financial/repayment features.
- [ ] `slice-04-preprocessing` — split data and build leakage-safe preprocessing.
- [ ] `slice-05-model-training` — train Logistic Regression, Random Forest, XGBoost/fallback.
- [ ] `slice-06-imbalance-evaluation` — apply weighting and record metrics.

## Depends on
Local copies of all seven Home Credit CSV files.

## Exit criteria
- Applicant-level modeling table exists locally.
- Raw and processed datasets remain untracked.
- At least two models run; target is three.
- Core evaluation metrics are recorded.
