# ADR-001 — Project Methodology Decisions

## 1. Applicant-level aggregation
**Decision:** Reduce historical tables to features keyed by `SK_ID_CURR` and keep one modeling row per applicant.

**Reason:** The prediction unit is the applicant; direct raw-row concatenation would duplicate applicants and create an unnecessarily large modeling table.

## 2. Default imbalance strategy
**Decision:** Use class weighting first. Use the appropriate positive-class weighting parameter for boosting.

**Reason:** It is simple and reproducible under the two-day constraint. SMOTE is optional only if time permits.

## 3. Evaluation protocol
**Decision:** Use stratified train/validation/test splits, training-only preprocessing, validation-only threshold tuning, and untouched final test evaluation.

**Reason:** This separates model fitting, decision-threshold selection, and final performance estimation.

## 4. Model scope
**Decision:** Compare Logistic Regression, Random Forest, and XGBoost, with HistGradientBoosting as the fallback.

**Reason:** This provides a transparent baseline, nonlinear tree model, and boosting model without spending the deadline on many algorithms.

## 5. Feature scope
**Decision:** Prefer compact, domain-driven financial and repayment features over polynomial expansion or automatic high-dimensional feature generation.

**Reason:** The project needs interpretable features and a reliable end-to-end result within two days.
