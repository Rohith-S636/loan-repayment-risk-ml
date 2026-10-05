# Slice 08 — Final Evaluation

## Module
`evaluate`

## What to build
Freeze the accepted model and validation-selected threshold, then produce final test-set metrics and figures.

## Acceptance criteria
- [x] Final test predictions use the accepted model/pipeline.
- [x] ROC-AUC and PR-AUC are recorded.
- [x] Precision, recall, F1, and confusion matrix are recorded at the selected threshold.
- [x] Final figures are saved.
- [x] Validation model/threshold selection and final test evaluation are clearly separated.
- [x] No tuning is performed after final test inspection.

## Important note
The pre-tuning evaluation remains a baseline experiment record. The accepted tuned model was evaluated after Slice 07 threshold selection. Final test results are ROC-AUC **0.778114**, PR-AUC **0.273767**, precision **0.256052**, recall **0.472910**, F1 **0.332225**, with TN **49,716**, FP **6,822**, FN **2,617**, and TP **2,348**.
