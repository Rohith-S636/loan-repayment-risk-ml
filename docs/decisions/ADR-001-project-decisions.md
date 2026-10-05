# ADR-001 — Project Methodology Decisions

## 1. Applicant-level aggregation
**Decision:** Reduce historical tables to features keyed by `SK_ID_CURR` and keep one modeling row per applicant.

**Reason:** The prediction unit is the applicant; direct raw-row concatenation would duplicate applicants and create an unnecessarily large modeling table.

## 2. Default imbalance strategy
**Decision:** Use class weighting first. Use the appropriate positive-class weighting parameter for boosting.

**Reason:** It is simple and reproducible under the two-day constraint. SMOTE is optional only if separately justified.

## 3. Evaluation protocol
**Decision:** Use stratified train/validation/test splits, training-only preprocessing, validation-only model and threshold selection, and untouched final test evaluation.

**Reason:** This separates model fitting, model selection, decision-threshold selection, and final performance estimation.

## 4. Model scope
**Decision:** Compare Logistic Regression, Random Forest, and XGBoost, with HistGradientBoosting as the fallback.

**Reason:** This provides a transparent baseline, nonlinear tree model, and boosting model without spending the deadline on many algorithms.

## 5. Feature scope
**Decision:** Prefer compact, domain-driven financial and repayment features over polynomial expansion or automatic high-dimensional feature generation.

**Reason:** The project needs interpretable features and a reliable end-to-end result within two days.

## 6. Controlled XGBoost tuning
**Decision:** Tune only XGBoost using a deterministic, limited validation experiment after baseline model comparison.

**Protocol:**
- 20 sampled configurations.
- Random seed 42.
- Tune tree complexity, learning rate, subsampling, regularization, and `scale_pos_weight`.
- Select by validation PR-AUC.
- Use validation ROC-AUC and F1 as tie-breakers.
- Never load or inspect the test split during tuning.
- Preserve the baseline model until the tuned candidate is compared against it.

**Reason:** XGBoost was the strongest baseline, so limited tuning provides a focused opportunity for improvement without turning the project into exhaustive hyperparameter optimization.

## 7. Post-tuning evaluation
**Decision:** If the tuned model is accepted, rerun threshold analysis on validation data and then perform one final test evaluation with the frozen model and threshold.

**Reason:** A changed model invalidates the old final test result as the final model's performance estimate. Test data must remain untouched until all model and threshold decisions are complete.
