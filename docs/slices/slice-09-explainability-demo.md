# Slice 09 — Explainability and Demo

## Module
`evaluate` / `app`

## Goal
Deliver the final explainable demonstration using the **accepted tuned XGBoost model** and the threshold frozen by Slice 07.

## Final configuration
- Model: `models/xgboost_tuned.joblib`
- Frozen threshold: **0.58**
- Model selection: controlled XGBoost tuning on the fixed training/validation split.
- Threshold selection: validation-only, maximizing F1.
- Final test evaluation: completed once in Slice 08.
- Test data is never used by the demo for training, tuning, threshold selection, or explainability.

## What to build

### 1. Permutation explainability
Generate validation-only permutation feature importance:

```powershell
python src\explainability.py
```

Default behavior:
- Uses `models/xgboost_tuned.joblib`.
- Samples up to 1,500 validation applicants.
- Uses average precision (PR-AUC) as the scoring function.
- Uses 3 permutation repeats.
- Fixed random seed: 42.
- Writes:
  - `results/explainability/permutation_importance.csv`
  - `results/explainability/explainability_metadata.json`

The untouched test split is never loaded.

### 2. Polished Streamlit demo

Launch from the repository root:

```powershell
python -m streamlit run app\app.py
```

The demo contains four views:

#### Risk Assessment
This is the primary live-demo screen.

- Selects a validation applicant as a reproducible historical credit profile.
- Accepts understandable current applicant/application details:
  - age
  - employment duration
  - number of children
  - family size
  - annual income
  - loan/credit amount
  - annual repayment/annuity
  - goods/purchase price
  - three normalized external credit indicators
- Derives the corresponding financial ratios used by Slice 03.
- Replaces the selected profile's current-application fields with the entered values.
- Retains historical bureau, previous-application, POS/CASH, installment, and credit-card aggregates from the selected validation profile.
- Runs the complete saved preprocessing pipeline and accepted tuned XGBoost model.
- Displays predicted repayment-difficulty probability.
- Applies the frozen **0.58** threshold to show LOW RISK / HIGH RISK.
- Shows the change from the selected applicant's baseline probability.
- Does not train, tune, or access the test split.

This design is intentional: a new applicant would not have all historical multi-table features available at manual entry time, so the demo does not pretend that those historical aggregates are user-entered. It uses a real validation-derived historical profile and makes the current application information interactive.

#### Model Performance
- Final test ROC-AUC: **0.778114**
- Final test PR-AUC: **0.273767**
- Final test precision: **0.256052**
- Final test recall: **0.472910**
- Final test F1: **0.332225**
- Final confusion matrix.
- Final ROC and precision-recall figures when Slice 08 artifacts are present.

#### Why This Prediction?
- Displays validation-only permutation feature importance.
- Shows the top features and their average-precision importance.
- Makes clear that importance is explanatory predictive signal, not causal evidence.

#### Threshold & Selection
- Compares Logistic Regression, Random Forest, baseline XGBoost, and tuned XGBoost.
- Shows why tuned XGBoost was accepted.
- Shows the 0.50 vs 0.58 validation operating point.
- Explains that threshold selection balances precision and recall using F1.

## Acceptance criteria
- [x] Final accepted tuned model is identified.
- [x] Frozen threshold is identified as 0.58.
- [x] Permutation importance generated locally with `src/explainability.py`.
- [x] Demo loads the final trained model and preprocessing pipeline.
- [x] Demo loads the frozen threshold.
- [x] Demo accepts understandable applicant/application details.
- [x] Entered application values are mapped into the model feature space.
- [x] Demo displays a risk probability.
- [x] Demo displays classification using the frozen threshold.
- [x] A reproducible validation-derived historical profile can be demonstrated.
- [x] UI clearly identifies the model and threshold.
- [x] Model performance and final test metrics are shown.
- [x] Threshold/model-selection reasoning is shown.
- [ ] Streamlit app starts successfully on the current project environment.

## Local verification checklist

From the project root:

```powershell
python src\explainability.py
python -m streamlit run app\app.py
```

Verify:
1. The app starts without traceback.
2. Risk Assessment clearly presents applicant/application fields rather than engineered ratios as the primary inputs.
3. Select a validation historical profile and submit the default values.
4. Record the baseline probability.
5. Change a meaningful application value, such as income, credit amount, annuity, goods price, or an external indicator, and submit again.
6. Confirm that the prediction is recalculated from the complete transformed row. The score may stay within the same tree decision region for some small changes; use a materially different scenario when demonstrating sensitivity.
7. Model Performance shows the Slice 08 final test metrics.
8. Why This Prediction? shows the generated permutation-importance table.
9. Threshold & Selection shows the accepted tuned model and threshold 0.58.
10. No page triggers training, tuning, or test-set evaluation.

## Out of scope
- Complex frontend frameworks.
- Cloud deployment.
- Additional model tuning.
- Post-test optimization.
- Re-training from the UI.
- Test-set explainability or test-set threshold selection.
