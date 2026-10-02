# Research — Loan Repayment Risk Prediction

## Problem
Predict whether a loan applicant is likely to experience repayment difficulty using current application information and historical credit/repayment behavior.

## Reference context
The reference study uses seven Home Credit source tables and explores multiple preprocessing, classifiers, imbalance methods, and exploratory analyses. This project uses the same problem/data family but intentionally narrows the implementation for a two-day deadline.

## Our focus
1. Applicant-level aggregation of historical records.
2. Domain-driven financial and repayment features.
3. Leakage-safe preprocessing.
4. Imbalance-aware model comparison.
5. Validation-based threshold analysis.
6. Interpretable feature importance.
7. A small working prediction demo.

## Research questions
- How strong is Logistic Regression as a baseline?
- Do nonlinear tree models improve predictive performance?
- How does class weighting affect minority-class detection?
- Does validation threshold selection change the precision/recall trade-off?
- Which engineered financial and repayment features are most important?

## Dataset
The seven local source tables are application_train, bureau, bureau_balance, previous_application, POS_CASH_balance, installments_payments, and credit_card_balance. The inspected raw collection is approximately 2.3 GB.

## Constraint
Only two days are available. Optional clustering, PCA/t-SNE, deep learning, polynomial expansion, exhaustive tuning, and cloud deployment are deferred.
