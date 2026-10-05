# Loan Repayment Risk Prediction Using Machine Learning

**Working title:** Loan Repayment Risk Prediction Using Machine Learning: An Imbalance-Aware and Interpretable Approach

A two-day ML mini-project for predicting loan repayment difficulty using the Home Credit multi-table dataset.

## Objective
Build a reproducible applicant-level pipeline combining current application data with historical credit and repayment behavior, handling class imbalance, comparing ML models, performing a controlled XGBoost tuning experiment, optimizing the decision threshold on validation data, and producing an interpretable demo.

## In scope
- All seven Home Credit source tables.
- Applicant-level historical aggregation.
- Compact financial and repayment features.
- Stratified train/validation/test split.
- Leakage-safe preprocessing.
- Logistic Regression, Random Forest, and XGBoost.
- Controlled, non-exhaustive XGBoost tuning.
- Class weighting as the default imbalance strategy.
- ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix.
- Validation-based model and threshold selection.
- Permutation feature importance.
- Polished Streamlit demo using the frozen final pipeline.
- Two-page report and presentation.

## Out of scope
Exhaustive hyperparameter search across all models, K-means, PCA/t-SNE, polynomial expansion, deep learning, complex frontend, cloud deployment, and SMOTE unless separately justified and time permits.

## Dataset
| Dataset | Rows | Columns |
|---|---:|---:|
| application_train.csv | 307,511 | 122 |
| bureau.csv | 1,716,428 | 17 |
| bureau_balance.csv | 27,299,925 | 3 |
| previous_application.csv | 1,670,214 | 37 |
| POS_CASH_balance.csv | 10,001,358 | 8 |
| installments_payments.csv | 13,605,401 | 8 |
| credit_card_balance.csv | 3,840,312 | 23 |

The raw CSVs are approximately 2.3 GB and are **not committed to GitHub**. Place them in `data/raw/`.

## Pipeline
```text
7 raw tables → validation → applicant-level aggregation
→ financial/repayment features → train/validation/test
→ leakage-safe preprocessing → baseline model comparison
→ controlled XGBoost tuning → validation model selection
→ threshold analysis → frozen final evaluation
→ permutation importance → Streamlit demo
```

## Final accepted model
Controlled tuning selected **XGBoost tuned** using validation PR-AUC.

Validation tuning:
- Baseline PR-AUC: **0.259062**
- Tuned PR-AUC: **0.261614**
- Baseline ROC-AUC: **0.772189**
- Tuned ROC-AUC: **0.774488**
- Tuned validation F1 at 0.50: **0.305564**

Slice 07 selected a frozen threshold of **0.58** using validation data only.

Final untouched test evaluation:
- ROC-AUC: **0.778114**
- PR-AUC: **0.273767**
- Precision: **0.256052**
- Recall: **0.472910**
- F1: **0.332225**

The final prediction pipeline uses:
- `models/xgboost_tuned.joblib`
- `models/preprocessor.joblib`
- `models/threshold.json`

## Model-selection protocol
The initial Slice 05 baseline selected XGBoost on validation performance. Slice 06A then performed a controlled 20-trial XGBoost tuning experiment using the fixed Slice 04 train/validation split. Selection used validation PR-AUC, with ROC-AUC and F1 tie-breakers. The test set was never used during tuning.

Slice 07 selected the operating threshold on validation data by maximizing F1. Slice 08 evaluated the frozen model and threshold exactly once on the untouched test split.

## Demo
Generate validation-only permutation importance:

```powershell
python src\explainability.py
```

Launch the polished Streamlit demo:

```powershell
python -m streamlit run app\app.py
```

The demo is a **single-page assessment dashboard**. It provides one continuous presentation flow:
1. **Applicant details** — age, employment, family, income, loan amount, annuity, and goods price.
2. **Risk score** — probability, LOW/HIGH decision, frozen threshold 0.58, and distance from threshold.
3. **Why this score?** — derived financial indicators plus individual XGBoost prediction contributions for the submitted applicant.
4. **Overall model signals** — validation-only permutation importance.
5. **Model evidence** — training/validation/test methodology, model comparison, final test metrics, confusion matrix, ROC/PR curves, and threshold analysis.

The page uses controlled synthetic historical-reference profiles so the complete multi-table feature representation can be demonstrated without selecting an arbitrary real applicant. The neutral profile is the validation median; the high-risk and low-risk stress-test references are aggregate profiles from the highest/lowest 1% of frozen-model validation scores. These profiles are demonstration controls, not additional trained models. External source indicators are not user-entered.

The UI never trains, tunes, or evaluates the test set.

## Reproducibility rules
- Fixed random seed: 42.
- One modeling row per `SK_ID_CURR`.
- Never use `TARGET` to construct predictors.
- Fit learned preprocessing on training data only.
- Tune/select models using training/validation data only.
- Select the classification threshold on validation data only.
- Keep the test set untouched until final evaluation.
- Never tune after inspecting final test results.
- Record actual experimental results; do not predeclare a winning tuned model.

## Local setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Controlled tuning
Run after the baseline pipeline is available:

```bash
python src/tune_xgboost.py
```

The experiment writes trial results under `results/tuning/` and saves the candidate model as `models/xgboost_tuned.joblib`.

See `docs/` for the project specification, architecture, research decisions, sprints, and implementation slices.
