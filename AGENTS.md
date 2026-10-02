# AGENTS.md

## Project
Loan Repayment Risk Prediction Using Machine Learning.

Read `docs/PRD.md` and `docs/ARCHITECTURE.md` before implementation.

## Hard scope
The project has a two-day deadline. Keep work focused on the complete ML pipeline:
seven-table aggregation, compact financial/repayment features, leakage-safe preprocessing, Logistic Regression, Random Forest, XGBoost/fallback, imbalance handling, evaluation, threshold analysis, feature importance, and a lightweight demo.

Do not add K-means, PCA/t-SNE, deep learning, complex frontend, cloud deployment, or exhaustive tuning unless explicitly requested.

## Data rules
- Raw CSVs stay local under `data/raw/` and must remain ignored by Git.
- Process large tables in chunks where practical.
- Final modeling data has one row per `SK_ID_CURR`.
- Never use `TARGET` to construct predictor features.
- Verify actual CSV columns before writing feature formulas.

## Experiment rules
- Fixed random seed.
- Stratified train/validation/test split.
- Fit learned preprocessing on training data only.
- Tune thresholds on validation data only.
- Use the test set only for final evaluation.
- Do not claim a model is better before the experiment shows it.

## Slice workflow
Work only on the active slice. Acceptance criteria are the test contract.
Use the slice ID in commits/PRs, e.g. `feat(slice-05): train baseline models`.

## Secrets
Never commit or expose secrets. There are no required secrets for the core ML pipeline.

## Decisions
Record non-obvious methodological decisions in `docs/decisions/ADR-001-project-decisions.md`.
