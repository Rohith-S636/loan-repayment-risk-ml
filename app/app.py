"""
Loan Repayment Risk — single-page academic demonstration.

Inference only:
- Uses the frozen tuned XGBoost model.
- Uses the frozen 0.58 decision threshold.
- Never trains, tunes, or evaluates the test set from the UI.
- Uses controlled synthetic historical-reference profiles rather than an
  arbitrary real validation applicant for the interactive demo.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import DMatrix

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "engineered_data.csv"
BUNDLE_PATH = ROOT / "models" / "preprocessing_bundle.joblib"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"
METRICS_PATH = ROOT / "results" / "metrics" / "final_test_metrics.json"
IMPORTANCE_PATH = ROOT / "results" / "explainability" / "permutation_importance.csv"
ROC_FIGURE = ROOT / "results" / "figures" / "final_test_roc_curve.png"
PR_FIGURE = ROOT / "results" / "figures" / "final_test_pr_curve.png"

FALLBACK_METRICS = {
    "model": "xgboost_tuned",
    "threshold": 0.58,
    "roc_auc": 0.778114,
    "pr_auc": 0.273767,
    "precision": 0.256052,
    "recall": 0.472910,
    "f1": 0.332225,
    "true_negative": 49716,
    "false_positive": 6822,
    "false_negative": 2617,
    "true_positive": 2348,
}

MODEL_COMPARISON = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Baseline XGBoost",
            "Tuned XGBoost",
        ],
        "Validation PR-AUC": [0.236606, 0.207645, 0.259062, 0.261614],
        "Validation ROC-AUC": [0.759910, 0.740854, 0.772189, 0.774488],
        "Validation F1": [0.269087, 0.277927, 0.291262, 0.305564],
    }
)

st.set_page_config(
    page_title="Loan Repayment Risk Assessment",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: #0b1020; }
    .hero {
        padding: 30px 34px; border-radius: 20px; margin-bottom: 22px;
        background: linear-gradient(135deg, #17223c 0%, #101729 100%);
        border: 1px solid #2b3b5d;
    }
    .hero h1 { margin: 0; font-size: 2.45rem; color: #f5f7ff; }
    .hero p { color: #aebbd6; margin: 9px 0 0; font-size: 1rem; }
    .section {
        padding: 20px 22px; border-radius: 16px; margin: 18px 0;
        background: #11182b; border: 1px solid #263550;
    }
    .risk-card {
        padding: 28px; border-radius: 18px; text-align: center;
        background: #141d32; border: 1px solid #30405f;
    }
    .risk-prob { font-size: 3.5rem; font-weight: 800; margin: 5px 0; }
    .risk-low { color: #57d7a2; }
    .risk-high { color: #ff7d8b; }
    .small-note { color: #8e9bb6; font-size: .88rem; }
    .explain-positive { color: #ff8a8a; }
    .explain-negative { color: #62d9ad; }
    div[data-testid="stMetric"] {
        background: #11182b; border: 1px solid #263550;
        padding: 12px; border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    required = [DATA_PATH, BUNDLE_PATH, THRESHOLD_PATH]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing local artifacts: " + ", ".join(missing))

    threshold = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    model_path = ROOT / threshold["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Frozen model not found: {model_path}")

    bundle = joblib.load(BUNDLE_PATH)
    model = joblib.load(model_path)
    data = pd.read_csv(DATA_PATH)

    feature_columns = bundle["feature_columns"]
    id_column = bundle["id_column"]
    validation_ids = set(bundle["split_ids"]["validation"])
    validation = data[data[id_column].isin(validation_ids)].sort_values(id_column)

    transformed_names = list(bundle["preprocessor"].get_feature_names_out())
    if len(transformed_names) != model.n_features_in_:
        raise ValueError("Model/preprocessor feature count mismatch.")

    # Build aggregate reference profiles from validation data. No applicant ID
    # is retained: each profile is the median feature vector of a validation
    # slice, so the demo does not expose or borrow one person's hidden history.
    reference_profiles = build_reference_profiles(model, bundle, validation, feature_columns)

    return {
        "model": model,
        "model_name": threshold["model"],
        "model_path": threshold["model_path"],
        "threshold": float(threshold["selected_threshold"]),
        "bundle": bundle,
        "data": data,
        "validation": validation,
        "feature_columns": feature_columns,
        "id_column": id_column,
        "transformed_names": transformed_names,
        "reference_profiles": reference_profiles,
    }


@st.cache_data(show_spinner="Preparing controlled demo reference profiles...")
def build_reference_profiles(model, bundle, validation: pd.DataFrame, feature_columns: list[str]):
    """Create synthetic reference histories from validation distributions.

    The neutral profile is the validation median. The high/low profiles are
    medians of the 1% validation slices with the highest/lowest frozen-model
    scores. They are aggregate profiles, not real applicants.
    """
    preprocessor = bundle["preprocessor"]
    X = preprocessor.transform(validation[feature_columns])
    scores = model.predict_proba(X)[:, 1]
    scored = validation[feature_columns].copy()
    scored["__demo_score"] = scores
    scored = scored.sort_values("__demo_score")

    n = max(100, int(len(scored) * 0.01))
    low = scored.head(n).drop(columns="__demo_score")
    high = scored.tail(n).drop(columns="__demo_score")

    def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
        values = {}
        for column in feature_columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                values[column] = float(series.median())
            else:
                mode = series.dropna().mode()
                values[column] = mode.iloc[0] if not mode.empty else np.nan
        return pd.DataFrame([values], columns=feature_columns)

    return {
        "neutral": aggregate(scored.drop(columns="__demo_score")),
        "high_risk": aggregate(high),
        "low_risk": aggregate(low),
    }


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return FALLBACK_METRICS.copy()


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return numerator / denominator


def finite_or(value, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if np.isfinite(value) else fallback


def build_demo_row(
    base: pd.DataFrame,
    *,
    age: float,
    employment_years: float,
    children: int,
    family_members: float,
    income: float,
    credit: float,
    annuity: float,
    goods_price: float,
) -> pd.DataFrame:
    """Overlay user-entered application details on a controlled reference profile."""
    row = base.copy()

    row["AMT_INCOME_TOTAL"] = income
    row["AMT_CREDIT"] = credit
    row["AMT_ANNUITY"] = annuity
    row["AMT_GOODS_PRICE"] = goods_price
    row["DAYS_BIRTH"] = -age * 365.25
    row["DAYS_EMPLOYED"] = -employment_years * 365.25
    row["CNT_CHILDREN"] = children
    row["CNT_FAM_MEMBERS"] = family_members

    row["APP_CREDIT_TO_INCOME"] = safe_ratio(credit, income)
    row["APP_ANNUITY_TO_INCOME"] = safe_ratio(annuity, income)
    row["APP_GOODS_PRICE_TO_INCOME"] = safe_ratio(goods_price, income)
    row["APP_CREDIT_TO_GOODS_PRICE"] = safe_ratio(credit, goods_price)
    row["APP_ANNUITY_TO_CREDIT"] = safe_ratio(annuity, credit)
    row["APP_AGE_YEARS"] = age
    row["APP_EMPLOYED_YEARS"] = employment_years
    row["APP_INCOME_PER_FAMILY_MEMBER"] = safe_ratio(
        income, family_members
    )
    return row


def predict_and_explain(artifacts: dict, row: pd.DataFrame):
    X = artifacts["bundle"]["preprocessor"].transform(
        row[artifacts["feature_columns"]]
    )
    model = artifacts["model"]

    probability = float(model.predict_proba(X)[:, 1][0])
    prediction = int(probability >= artifacts["threshold"])

    # Native XGBoost contributions explain the individual prediction in
    # log-odds space. Last value is the bias/base contribution.
    booster = model.get_booster()
    contributions = booster.predict(
        DMatrix(X),
        pred_contribs=True,
        validate_features=False,
    )[0]

    names = np.asarray(artifacts["transformed_names"])
    feature_contrib = pd.DataFrame(
        {
            "feature": names,
            "contribution": contributions[:-1],
        }
    )
    feature_contrib["abs_contribution"] = feature_contrib["contribution"].abs()
    feature_contrib = feature_contrib.sort_values(
        "abs_contribution", ascending=False
    )

    return probability, prediction, feature_contrib


def readable_feature(name: str) -> str:
    name = (
        name.replace("num__", "")
        .replace("cat__", "")
        .replace("missingindicator_", "Missing: ")
    )
    replacements = {
        "AMT_INCOME_TOTAL": "Annual income",
        "AMT_CREDIT": "Loan / credit amount",
        "AMT_ANNUITY": "Annual repayment / annuity",
        "AMT_GOODS_PRICE": "Goods / purchase price",
        "DAYS_BIRTH": "Age",
        "DAYS_EMPLOYED": "Employment duration",
        "CNT_CHILDREN": "Number of children",
        "CNT_FAM_MEMBERS": "Family members",
        "APP_CREDIT_TO_INCOME": "Loan-to-income ratio",
        "APP_ANNUITY_TO_INCOME": "Annuity-to-income ratio",
        "APP_GOODS_PRICE_TO_INCOME": "Purchase-price-to-income ratio",
        "APP_CREDIT_TO_GOODS_PRICE": "Loan-to-goods-price ratio",
        "APP_ANNUITY_TO_CREDIT": "Annuity-to-loan ratio",
        "APP_AGE_YEARS": "Age (derived)",
        "APP_EMPLOYED_YEARS": "Employment duration (derived)",
        "APP_INCOME_PER_FAMILY_MEMBER": "Income per family member",
        "PREV_REFUSAL_RATIO": "Previous application refusal ratio",
        "BUREAU_DEBT_TO_CREDIT": "Bureau debt-to-credit ratio",
        "INST_AVG_DELAY_POSITIVE": "Average installment delay",
        "CC_HIGH_UTILIZATION_FLAG": "High credit-card utilization",
    }
    return replacements.get(name, name.replace("_", " ").title())


def render_local_explanation(
    contribution_df: pd.DataFrame,
    row: pd.DataFrame,
) -> None:
    st.markdown("### Why did the model give this risk score?")

    top = contribution_df.head(10).copy()
    top["Feature"] = top["feature"].map(readable_feature)
    top["Effect"] = np.where(
        top["contribution"] >= 0,
        "Pushes risk higher",
        "Pushes risk lower",
    )
    top["Contribution"] = top["contribution"]

    st.caption(
        "The chart shows the strongest individual model contributions for this "
        "applicant. Positive contributions push the model toward repayment "
        "difficulty; negative contributions push it toward lower risk."
    )
    st.bar_chart(
        top.set_index("Feature")["Contribution"],
        horizontal=True,
    )

    display = top[["Feature", "Effect", "Contribution"]].copy()
    display["Contribution"] = display["Contribution"].map(lambda x: f"{x:+.3f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.info(
        "These are model contributions, not causal claims. The explanation "
        "reflects the complete applicant representation, including the selected "
        "synthetic historical-reference profile."
    )


def render_input_summary(
    age, employment_years, children, family_members,
    income, credit, annuity, goods_price,
):
    st.markdown("### What the entered details produce")
    ratios = pd.DataFrame(
        {
            "Derived indicator": [
                "Loan / income",
                "Annuity / income",
                "Goods price / income",
                "Loan / goods price",
                "Annuity / loan",
                "Income / family member",
            ],
            "Value": [
                safe_ratio(credit, income),
                safe_ratio(annuity, income),
                safe_ratio(goods_price, income),
                safe_ratio(credit, goods_price),
                safe_ratio(annuity, credit),
                safe_ratio(income, family_members),
            ],
        }
    )
    ratios["Value"] = ratios["Value"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    st.dataframe(ratios, use_container_width=True, hide_index=True)


def render_model_evidence(artifacts: dict, metrics: dict) -> None:
    st.markdown("---")
    st.header("How the model was trained and evaluated")

    st.markdown(
        """
        **Data:** 7 Home Credit tables → applicant-level historical aggregation
        → domain feature engineering → leakage-safe preprocessing.

        **Split:** 60% training / 20% validation / 20% untouched test,
        stratified with random seed 42.

        **Models:** Logistic Regression, Random Forest and XGBoost.
        A controlled 20-trial XGBoost experiment was then performed using
        training/validation data only.

        **Selection:** validation PR-AUC, with ROC-AUC/F1 tie-breakers.
        The operating threshold was then selected on validation data by
        maximizing F1.

        **Final model:** tuned XGBoost with frozen threshold **0.58**.
        The test set was evaluated only after model and threshold selection.
        """
    )

    train_rows = artifacts["bundle"]["split_sizes"]["train"]
    valid_rows = artifacts["bundle"]["split_sizes"]["validation"]
    test_rows = artifacts["bundle"]["split_sizes"]["test"]

    c = st.columns(4)
    c[0].metric("Training rows", f"{train_rows:,}")
    c[1].metric("Validation rows", f"{valid_rows:,}")
    c[2].metric("Test rows", f"{test_rows:,}")
    c[3].metric("Model features", len(artifacts["feature_columns"]))

    st.markdown("### Validation model comparison")
    st.dataframe(MODEL_COMPARISON, use_container_width=True, hide_index=True)

    st.markdown("### Final untouched test results")
    c = st.columns(5)
    for col, label, key in zip(
        c,
        ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"],
        ["roc_auc", "pr_auc", "precision", "recall", "f1"],
    ):
        col.metric(label, f"{metrics[key]:.4f}")

    cm = pd.DataFrame(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ],
        index=["Actual Low Risk", "Actual High Risk"],
        columns=["Predicted Low Risk", "Predicted High Risk"],
    )
    st.markdown("### Confusion matrix")
    st.dataframe(cm, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if ROC_FIGURE.exists():
            st.image(str(ROC_FIGURE), caption="Final test ROC curve")
    with c2:
        if PR_FIGURE.exists():
            st.image(str(PR_FIGURE), caption="Final test precision-recall curve")

    st.markdown("### Decision threshold")
    threshold_table = pd.DataFrame(
        {
            "Operating point": ["Default 0.50", "Selected 0.58"],
            "Precision": [0.207335, 0.251090],
            "Recall": [0.580665, 0.475730],
            "F1": [0.305564, 0.328695],
        }
    )
    st.dataframe(threshold_table, use_container_width=True, hide_index=True)

    st.caption(
        "Final test results were produced once. No tuning or threshold "
        "selection was performed after inspecting the test set."
    )


def render_global_explainability() -> None:
    if not IMPORTANCE_PATH.exists():
        return

    st.markdown("### Overall model signals")
    importance = pd.read_csv(IMPORTANCE_PATH).sort_values(
        "importance_mean", ascending=False
    )
    top = importance.head(10).copy()
    top["Feature"] = top["feature"].map(readable_feature)
    st.bar_chart(
        top.set_index("Feature")["importance_mean"],
        horizontal=True,
    )
    st.caption(
        "Validation permutation importance shows which features most affect "
        "overall validation PR-AUC. It complements the individual prediction "
        "explanation above and is not causal evidence."
    )


def main() -> None:
    try:
        artifacts = load_artifacts()
    except Exception as exc:
        st.error("The demo cannot start because required ML artifacts are missing.")
        st.code(str(exc))
        st.stop()

    metrics = load_metrics()

    st.markdown(
        f"""
        <div class="hero">
            <h1>Loan Repayment Risk Assessment</h1>
            <p>
                Enter borrower details → generate risk score → understand the
                decision → review how the model was trained and evaluated.
                Final model: <b>{artifacts['model_name']}</b> · threshold
                <b>{artifacts['threshold']:.2f}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "This is an academic demonstration. The eight visible application "
        "inputs are combined with a controlled synthetic historical-reference "
        "profile derived from validation-data distributions. No real applicant "
        "ID or individual's hidden bureau/payment history is used."
    )

    st.header("1. Enter applicant details")

    reference_labels = {
        "neutral": "Neutral historical reference",
        "high_risk": "High-risk stress-test reference",
        "low_risk": "Low-risk reference",
    }
    reference_mode = st.selectbox(
        "Historical feature reference",
        list(reference_labels),
        format_func=lambda key: reference_labels[key],
        index=0,
        help=(
            "The trained model uses historical credit/payment features that a "
            "new applicant cannot reasonably type into this demo. These controls "
            "use aggregate validation profiles rather than a real applicant."
        ),
    )
    st.caption(
        "Only the application details below are entered by the user. Historical "
        "bureau/payment variables are supplied by the selected synthetic reference "
        "profile. Use the high-risk stress-test reference for the academic demo."
    )

    reference_base = artifacts["reference_profiles"][reference_mode].copy()

    # Use representative, easy-to-explain defaults rather than values copied
    # from an individual applicant.
    default_age = 35.0
    default_employment = 5.0
    default_children = 1
    default_family = 3.0
    default_income = 300000.0
    default_credit = 500000.0
    default_annuity = 25000.0
    default_goods = 450000.0

    with st.form("risk_assessment_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age (years)", 18.0, 75.0, default_age, 1.0)
        with c2:
            employment_years = st.number_input(
                "Employment duration (years)", 0.0, 45.0,
                default_employment, 0.5,
            )
        with c3:
            children = st.number_input(
                "Number of children", 0, 10, default_children, 1
            )
        with c4:
            family_members = st.number_input(
                "Family members", 1.0, 15.0,
                default_family, 1.0,
            )

        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input(
                "Annual income (₹)", min_value=10000.0,
                value=default_income, step=10000.0, format="%.0f",
            )
            credit = st.number_input(
                "Loan / credit amount (₹)", min_value=10000.0,
                value=default_credit, step=10000.0, format="%.0f",
            )
        with c2:
            annuity = st.number_input(
                "Annual repayment / annuity (₹)", min_value=1000.0,
                value=default_annuity, step=1000.0, format="%.0f",
            )
            goods_price = st.number_input(
                "Goods / purchase price (₹)", min_value=10000.0,
                value=default_goods, step=10000.0, format="%.0f",
            )

        submitted = st.form_submit_button(
            "ASSESS REPAYMENT RISK",
            type="primary",
            use_container_width=True,
        )

    if "assessment" not in st.session_state:
        st.session_state["assessment"] = None

    if submitted:
        row = build_demo_row(
            reference_base,
            age=age,
            employment_years=employment_years,
            children=children,
            family_members=family_members,
            income=income,
            credit=credit,
            annuity=annuity,
            goods_price=goods_price,
        )
        probability, prediction, contributions = predict_and_explain(
            artifacts, row
        )
        st.session_state["assessment"] = {
            "row": row,
            "probability": probability,
            "prediction": prediction,
            "contributions": contributions,
            "reference_mode": reference_mode,
        }

    assessment = st.session_state["assessment"]

    if assessment is not None:
        probability = assessment["probability"]
        prediction = assessment["prediction"]
        contributions = assessment["contributions"]

        st.markdown("---")
        st.header("2. Risk score and decision")

        risk_class = "risk-high" if prediction else "risk-low"
        label = "HIGH RISK" if prediction else "LOW RISK"

        left, right = st.columns([1.15, 1])
        with left:
            st.markdown(
                f"""
                <div class="risk-card">
                    <div class="small-note">
                        Predicted probability of repayment difficulty
                    </div>
                    <div class="risk-prob {risk_class}">{probability:.1%}</div>
                    <div class="{risk_class}"
                         style="font-size:1.35rem;font-weight:800">
                        {label}
                    </div>
                    <div class="small-note">
                        Frozen decision threshold: {artifacts['threshold']:.0%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.metric("Accepted model", artifacts["model_name"])
            st.metric(
                "Distance from threshold",
                f"{probability - artifacts['threshold']:+.1%}",
            )
            st.caption(
                "The probability is a model score for repayment difficulty, "
                "not a guaranteed outcome or a calibrated probability claim."
            )

        render_input_summary(
            age, employment_years, children, family_members,
            income, credit, annuity, goods_price,
        )
        render_local_explanation(contributions, assessment["row"])

        render_global_explainability()

    else:
        st.markdown(
            """
            <div class="section">
                <b>Ready for assessment.</b><br>
                Enter the borrower details above and select
                <b>ASSESS REPAYMENT RISK</b>. The score, local explanation,
                derived indicators, and model evidence will appear below.
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_model_evidence(artifacts, metrics)

    st.markdown("---")
    st.caption(
        "Single-page inference demo · frozen tuned XGBoost · threshold 0.58 · "
        "no training, tuning, or test-set evaluation occurs in the UI."
    )


if __name__ == "__main__":
    main()
