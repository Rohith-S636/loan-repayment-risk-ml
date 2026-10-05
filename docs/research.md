# Research — Loan Repayment Risk Prediction

## Problem
Predict whether a loan applicant is likely to experience repayment difficulty using current application information and historical credit/repayment behavior.

## Reference context
The reference study uses seven Home Credit source tables and explores multiple preprocessing, classifiers, imbalance methods, and exploratory analyses. This project uses the same problem/data family but intentionally narrows and strengthens the implementation for a two-day deadline.

## Our focus
1. Applicant-level aggregation of historical records.
2. Domain-driven financial and repayment features.
3. Leakage-safe preprocessing.
4. Imbalance-aware model comparison.
5. Controlled XGBoost tuning using validation data only.
6. Validation-based threshold analysis.
7. Interpretable feature importance.
8. A small working prediction demo.

## Research questions
- How strong is Logistic Regression as a baseline?
- Do nonlinear tree models improve predictive performance?
- Does XGBoost outperform the simpler candidate models on PR-AUC and ROC-AUC?
- Can a limited XGBoost tuning experiment improve validation PR-AUC without introducing test-set leakage?
- How does class weighting affect minority-class detection?
- Does validation threshold selection change the precision/recall trade-off?
- Which engineered financial and repayment features are most important?

## Baseline
The initial XGBoost configuration achieved:
- Validation ROC-AUC: **0.772189**
- Validation PR-AUC: **0.259062**
- Validation F1 at threshold 0.50: **0.291262**

These are the baseline values against which the controlled tuning experiment is compared.

## Controlled tuning protocol
Only XGBoost is tuned because it was the strongest baseline model. The experiment samples 20 configurations with fixed random seed 42 and varies tree complexity, learning rate, sampling, regularization, and positive-class weighting.

The primary selection metric is validation PR-AUC. Validation ROC-AUC and F1 are secondary tie-breakers. The existing Slice 04 train/validation IDs and preprocessing are reused. The test split is never loaded or used during tuning.

The tuned model was accepted after improving validation PR-AUC to **0.261614** and validation ROC-AUC to **0.774488**. Threshold analysis was rerun on validation data, selecting **0.58** by maximizing F1, and the final test set was evaluated exactly once with the frozen model and threshold. Final test ROC-AUC was **0.778114**, PR-AUC **0.273767**, precision **0.256052**, recall **0.472910**, and F1 **0.332225**.

## Dataset
The seven local source tables are application_train, bureau, bureau_balance, previous_application, POS_CASH_balance, installments_payments, and credit_card_balance. The inspected raw collection is approximately 2.3 GB.

## Constraint
The tuning experiment is deliberately limited rather than exhaustive. Optional clustering, PCA/t-SNE, deep learning, polynomial expansion, full hyperparameter optimization across every model, and cloud deployment remain out of scope.
