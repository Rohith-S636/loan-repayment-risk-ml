# Slice 04 — Preprocessing

## Module
`preprocessing`

## What to build
Create a stratified train/validation/test split and reusable preprocessing pipeline.

## Acceptance criteria
- [ ] Split is stratified by `TARGET`.
- [ ] Random seed is fixed.
- [ ] Numeric missing values are handled.
- [ ] Categorical values are encoded.
- [ ] Learned preprocessing is fitted on training data only.
- [ ] Validation/test are transformed without refitting.

## Out of scope
Target leakage through validation/test statistics.
