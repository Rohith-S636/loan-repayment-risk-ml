# Slice 09 — Explainability and Demo

## Module
\x60evaluate\x60 / \x60app\x60

## Goal
Deliver the final explainable demonstration using the **accepted tuned XGBoost model** and the threshold frozen by Slice 07.

## Final configuration
- Model: \x60models/xgboost_tuned.joblib\x60
- Frozen threshold: **0.58**
- Model selection: controlled XGBoost tuning on the fixed training/validation split.
- Threshold selection: validation-only, maximizing F1.
- Final test evaluation: completed once in Slice 08.
- Test data is never used by the demo for training, tuning, threshold selection, or explainability.

## What to build

### 1. Permutation explainability
Generate validation-only permutation feature importance:

\x60\x60\x60powershell
python src\\explainability.py
\x60\x60\x60

Default behavior:
- Uses \x60models/xgboost_tuned.joblib\x60.
- Samples up to 1,500 validation applicants.
- Uses average precision (PR-AUC) as the scoring function.
- Uses 3 permutation repeats.
- Fixed random seed: 42.
- Writes:
  - \x60results/explainability/permutation_importance.csv\x60
  - \x60results/explainability/explainability_metadata.json\x60

The untouched test split is never loaded.

### 2. Polished Streamlit demo

Launch from the repository root:

\x60\x60\x60powershell
streamlit run app\\app.py
\x60\x60\x60

The demo contains four views:

#### Risk Predictor
- Reproducible validation-applicant selection.
- Scenario explorer for key interpretable engineered features.
- Predicted repayment-difficulty probability.
- Low Risk / High Risk decision using the frozen **0.58** threshold.
- Model and threshold shown explicitly.

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
- Makes clear that importance is explanatory signal, not causal evidence.

#### Threshold & Selection
- Compares Logistic Regression, Random Forest, baseline XGBoost, and tuned XGBoost.
- Shows why tuned XGBoost was accepted.
- Shows the 0.50 vs 0.58 validation operating point.
- Explains that threshold selection balances precision and recall using F1.

## Acceptance criteria
- [x] Final accepted tuned model is identified.
- [x] Frozen threshold is identified as 0.58.
- [ ] Permutation importance generated locally with \x60src/explainability.py\x60.
- [ ] Demo loads the final trained model and preprocessing pipeline.
- [ ] Demo loads the frozen threshold.
- [ ] Demo displays a risk probability.
- [ ] Demo displays classification using the frozen threshold.
- [ ] A reproducible validation applicant can be demonstrated.
- [ ] UI clearly identifies the model and threshold.
- [ ] Model performance and final test metrics are shown.
- [ ] Threshold/model-selection reasoning is shown.
- [ ] Streamlit app starts successfully on the local project environment.

## Local verification checklist

From the project root:

\x60\x60\x60powershell
python src\\explainability.py
streamlit run app\\app.py
\x60\x60\x60

Verify:
1. The app starts without traceback.
2. Risk Predictor produces a probability and class.
3. Changing scenario values changes the prediction when the selected profile is sensitive to those features.
4. Model Performance shows the Slice 08 final test metrics.
5. Why This Prediction? shows the generated permutation-importance table.
6. Threshold & Selection shows the accepted tuned model and threshold 0.58.
7. No page triggers training, tuning, or test-set evaluation.

## Out of scope
- Complex frontend frameworks.
- Cloud deployment.
- Additional model tuning.
- Post-test optimization.
- Re-training from the UI.
- Test-set explainability or test-set threshold selection.
