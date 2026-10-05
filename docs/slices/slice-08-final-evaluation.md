# Slice 08 — Final Evaluation

## Module
`evaluate`

## What to build
Freeze the accepted model and validation-selected threshold, then produce final test-set metrics and figures.

## Acceptance criteria
- [ ] Final test predictions use the accepted model/pipeline.
- [ ] ROC-AUC and PR-AUC are recorded.
- [ ] Precision, recall, F1, and confusion matrix are recorded at the selected threshold.
- [ ] Final figures are saved.
- [ ] Validation model/threshold selection and final test evaluation are clearly separated.
- [ ] No tuning is performed after final test inspection.

## Important note
The Slice 08 evaluation completed before the controlled tuning experiment is a valid baseline evaluation. If Slice 06A accepts a tuned model, Slice 07 and Slice 08 must be rerun so the reported final results correspond to the accepted model.
