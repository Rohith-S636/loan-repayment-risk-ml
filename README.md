# Loan Repayment Risk Prediction Using Machine Learning (An Imbalance-Aware and Interpretable Approach)

A ML mini-project for predicting loan repayment difficulty using the Home Credit multi-table dataset.

## Submission artifacts
- [Two-page project report](./Project%20Report.pdf)
- [Project presentation](./Project%20PPT.pdf)
## Presented By
- **Rohith G S (PES2UG24AM138)**
- **Nakshathira B (PES2UG24AM096)**

The report and presentation use the recorded experiment results in this repository. Raw Home
Credit CSV files are intentionally not included.

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
- `models/preprocessing_bundle.joblib`
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

## Clean-clone setup and complete reproduction

After cloning the repository,download the raw data from the [Home Credit Default Risk Kaggle Competition](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place the seven original Home Credit CSV files in
`data/raw/`. The raw files are ignored by Git and must be obtained separately. From
the repository root, run the pipeline in this order:

```powershell
# 1. Validate the seven local source tables and their relationships
python src\inspect_datasets.py

# 2. Aggregate historical tables to one row per SK_ID_CURR
python src\aggregate_history.py

# 3. Build the compact financial and repayment feature table
python src\feature_engineering.py

# 4. Create the fixed stratified train/validation/test split and fit preprocessing on train only
python src\preprocess.py

# 5. Train and compare the Logistic Regression, Random Forest, and baseline XGBoost models
python src\train.py

# 6. Evaluate the selected imbalance-handling approach
python src\evaluate_imbalance.py

# 7. Run the deterministic 20-trial XGBoost validation-only tuning experiment
python src\tune_xgboost.py

# 8. Select the operating threshold on validation data only
python src\threshold_analysis.py

# 9. Evaluate the frozen model and threshold once on the untouched test set
python src\final_evaluation.py

# 10. Generate validation-only permutation feature importance
python src\explainability.py

# 11. Launch the inference-only Streamlit demonstration
python -m streamlit run app\app.py
```

The first ten commands regenerate the required intermediate and final artifacts before the
demo starts. The final evaluation must not be rerun with a changed model, split, or threshold
after inspecting test results.

The clean-clone demo uses these lightweight final artifacts:

- `models/xgboost_tuned.joblib`
- `models/preprocessing_bundle.joblib`
- `models/threshold.json`
- `data/processed/engineered_data.csv`
- `results/figures/final_test_confusion_matrix.png`
- `results/figures/final_test_roc_curve.png`
- `results/figures/final_test_pr_curve.png`
- `results/metrics/final_test_metrics.csv`
- `results/metrics/final_test_metrics.json`
- `results/metrics/final_test_confusion_matrix.csv`
- `results/explainability/permutation_importance.csv`

Large raw inputs and intermediate generated files remain ignored. Running the commands above
recreates them locally when the raw data is available.

See `docs/` for the project specification, architecture, research decisions, sprints, and implementation slices.
