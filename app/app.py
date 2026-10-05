"""
Loan Repayment Risk — Slice 09 Streamlit demonstration.

The app is inference-only. It uses the frozen accepted model and threshold
recorded by Slice 07. The prediction page accepts understandable applicant
details, derives the corresponding application features, and combines them
with a selected validation applicant's historical credit profile.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

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

st.set_page_config(
    page_title="Loan Repayment Risk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #0b1020; }
    [data-testid="stSidebar"] { background: #11182b; }
    .hero {
        padding: 28px 30px; border-radius: 18px; margin-bottom: 20px;
        background: linear-gradient(135deg, #17223c 0%, #11182b 100%);
        border: 1px solid #293756;
    }
    .hero h1 { margin: 0; font-size: 2.35rem; color: #f5f7ff; }
    .hero p { color: #aebbd6; margin: 8px 0 0; font-size: 1rem; }
    .section-card {
        padding: 18px; border-radius: 16px; margin-bottom: 14px;
        background: #11182b; border: 1px solid #263550;
    }
    .risk-card {
        padding: 28px; border-radius: 18px; text-align: center;
        background: #141d32; border: 1px solid #30405f;
    }
    .risk-prob { font-size: 3.1rem; font-weight: 800; margin: 8px 0; }
    .risk-low { color: #57d7a2; }
    .risk-high { color: #ff7d8b; }
    .small-note { color: #8e9bb6; font-size: .88rem; }
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
        raise FileNotFoundError(
            "Missing local pipeline artifacts: " + ", ".join(missing)
        )

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
    validation = data[data[id_column].isin(validation_ids)].copy()
    validation = validation.sort_values(id_column)

    if len(validation) != bundle["split_sizes"]["validation"]:
        raise ValueError("Validation rows do not match the saved Slice 04 split.")

    transformed_names = list(bundle["preprocessor"].get_feature_names_out())
    if len(transformed_names) != model.n_features_in_:
        raise ValueError("Model/preprocessor feature count mismatch.")

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
    }


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return FALLBACK_METRICS.copy()


def get_prediction(artifacts: dict, row: pd.DataFrame) -> tuple[float, int]:
    X = artifacts["bundle"]["preprocessor"].transform(
        row[artifacts["feature_columns"]]
    )
    probability = float(artifacts["model"].predict_proba(X)[:, 1][0])
    prediction = int(probability >= artifacts["threshold"])
    return probability, prediction


def render_header(artifacts: dict) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>Loan Repayment Risk Assessment</h1>
            <p>
                AI-based repayment-difficulty assessment using tuned XGBoost
                · frozen decision threshold {artifacts['threshold']:.2f}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return numerator / denominator


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
    ext_source_1: float,
    ext_source_2: float,
    ext_source_3: float,
) -> pd.DataFrame:
    """Apply user-facing application details to a validation-derived profile."""
    row = base.copy()

    row["AMT_INCOME_TOTAL"] = income
    row["AMT_CREDIT"] = credit
    row["AMT_ANNUITY"] = annuity
    row["AMT_GOODS_PRICE"] = goods_price
    row["DAYS_BIRTH"] = -age * 365.25
    row["DAYS_EMPLOYED"] = -employment_years * 365.25
    row["CNT_CHILDREN"] = children
    row["CNT_FAM_MEMBERS"] = family_members
    row["EXT_SOURCE_1"] = ext_source_1
    row["EXT_SOURCE_2"] = ext_source_2
    row["EXT_SOURCE_3"] = ext_source_3

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


def render_risk_result(
    probability: float,
    prediction: int,
    threshold: float,
    baseline_probability: float,
    model_name: str,
) -> None:
    risk_class = "risk-high" if prediction else "risk-low"
    label = "HIGH RISK" if prediction else "LOW RISK"
    delta = probability - baseline_probability

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
                     style="font-size:1.25rem;font-weight:700">
                    {label}
                </div>
                <div class="small-note">
                    Decision threshold: {threshold:.0%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.metric(
            "Change from selected applicant baseline",
            f"{delta:+.1%}",
        )
        st.metric("Model", model_name)
        st.caption(
            "The score is generated by the frozen preprocessing pipeline "
            "and tuned XGBoost model. No training or tuning occurs in the app."
        )


def render_predictor(artifacts: dict) -> None:
    st.subheader("Applicant Risk Assessment")
    st.caption(
        "Enter understandable borrower/application details. The app derives "
        "the corresponding financial features and combines them with the "
        "selected validation applicant's historical credit profile."
    )

    validation = artifacts["validation"]
    id_column = artifacts["id_column"]

    profile_ids = validation[id_column].astype(int).tolist()
    selected_id = st.selectbox(
        "Demonstration historical profile",
        profile_ids,
        index=0,
        format_func=lambda x: f"Validation applicant {x:,}",
    )
    base = validation[validation[id_column] == selected_id].copy()

    st.info(
        "Demo design: current application details are entered below, while "
        "historical bureau/payment features come from the selected validation "
        "profile. This lets the complete trained 256-feature pipeline be "
        "demonstrated without inventing unavailable credit history."
    )

    source = base.iloc[0]

    preset = st.selectbox(
        "Optional demo starting point",
        ["Custom", "Lower-burden example", "Higher-burden example"],
        help="Presets only change the starting values. You can edit every field.",
    )

    default_age = float(np.clip(source.get("APP_AGE_YEARS", 35.0), 18, 75))
    default_employment = float(
        np.clip(source.get("APP_EMPLOYED_YEARS", 5.0), 0, 45)
    )
    default_children = int(np.clip(source.get("CNT_CHILDREN", 0), 0, 10))
    default_family = float(
        np.clip(source.get("CNT_FAM_MEMBERS", max(default_children + 1, 1)), 1, 15)
    )
    default_income = float(max(source.get("AMT_INCOME_TOTAL", 250000.0), 1))
    default_credit = float(max(source.get("AMT_CREDIT", 500000.0), 1))
    default_annuity = float(max(source.get("AMT_ANNUITY", 25000.0), 1))
    default_goods = float(max(source.get("AMT_GOODS_PRICE", default_credit), 1))

    if preset == "Lower-burden example":
        default_income = max(default_income * 1.75, 500000.0)
        default_credit = max(default_credit * 0.70, 100000.0)
        default_annuity = max(default_annuity * 0.75, 5000.0)
    elif preset == "Higher-burden example":
        default_income = max(default_income * 0.55, 50000.0)
        default_credit = max(default_credit * 1.45, 100000.0)
        default_annuity = max(default_annuity * 1.30, 5000.0)

    with st.form("risk_assessment_form"):
        st.markdown("### 1. Applicant Information")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input(
                "Age (years)", 18.0, 75.0, default_age, 1.0
            )
        with c2:
            employment_years = st.number_input(
                "Employment duration (years)",
                0.0, 45.0, default_employment, 0.5,
            )
        with c3:
            children = st.number_input(
                "Number of children", 0, 10, default_children, 1
            )
        with c4:
            family_members = st.number_input(
                "Family members", 1.0, 15.0, default_family, 1.0
            )

        st.markdown("### 2. Financial Information")
        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input(
                "Annual income (₹)",
                min_value=10000.0,
                value=default_income,
                step=10000.0,
                format="%.0f",
            )
            credit = st.number_input(
                "Loan / credit amount (₹)",
                min_value=10000.0,
                value=default_credit,
                step=10000.0,
                format="%.0f",
            )
        with c2:
            annuity = st.number_input(
                "Annual repayment / annuity (₹)",
                min_value=1000.0,
                value=default_annuity,
                step=1000.0,
                format="%.0f",
            )
            goods_price = st.number_input(
                "Goods / purchase price (₹)",
                min_value=10000.0,
                value=default_goods,
                step=10000.0,
                format="%.0f",
            )

        st.markdown("### 3. External Credit Indicators")
        st.caption(
            "These model inputs are represented on a normalized 0–1 scale in "
            "the source data; they are not claimed to be consumer credit scores."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            ext_source_1 = st.slider(
                "External indicator 1",
                0.0, 1.0,
                float(np.clip(source.get("EXT_SOURCE_1", 0.5), 0, 1)),
                0.01,
            )
        with c2:
            ext_source_2 = st.slider(
                "External indicator 2",
                0.0, 1.0,
                float(np.clip(source.get("EXT_SOURCE_2", 0.5), 0, 1)),
                0.01,
            )
        with c3:
            ext_source_3 = st.slider(
                "External indicator 3",
                0.0, 1.0,
                float(np.clip(source.get("EXT_SOURCE_3", 0.5), 0, 1)),
                0.01,
            )

        st.markdown("### 4. Assessment")
        submitted = st.form_submit_button(
            "Assess Repayment Risk",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        st.markdown(
            """
            <div class="section-card">
                <strong>Ready for assessment</strong><br>
                Enter the borrower details and click <b>Assess Repayment Risk</b>
                to run the frozen model.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    row = build_demo_row(
        base,
        age=age,
        employment_years=employment_years,
        children=children,
        family_members=family_members,
        income=income,
        credit=credit,
        annuity=annuity,
        goods_price=goods_price,
        ext_source_1=ext_source_1,
        ext_source_2=ext_source_2,
        ext_source_3=ext_source_3,
    )

    probability, prediction = get_prediction(artifacts, row)
    baseline_probability, _ = get_prediction(artifacts, base)

    st.markdown("### Assessment Result")
    render_risk_result(
        probability,
        prediction,
        artifacts["threshold"],
        baseline_probability,
        artifacts["model_name"],
    )

    st.markdown("### Derived financial indicators")
    ratios = pd.DataFrame(
        {
            "Indicator": [
                "Credit / income",
                "Annuity / income",
                "Credit / goods price",
                "Annuity / credit",
                "Income / family member",
            ],
            "Value": [
                safe_ratio(credit, income),
                safe_ratio(annuity, income),
                safe_ratio(credit, goods_price),
                safe_ratio(annuity, credit),
                safe_ratio(income, family_members),
            ],
        }
    )
    ratios["Value"] = ratios["Value"].map(
        lambda value: f"{value:.3f}" if pd.notna(value) else "N/A"
    )
    st.dataframe(ratios, use_container_width=True, hide_index=True)

    with st.expander("Demo methodology"):
        st.write(
            "The selected validation applicant supplies the historical "
            "bureau, previous-application, installment, POS/CASH, and "
            "credit-card aggregates. The entered application details replace "
            "the current-application values and their derived ratios. The "
            "frozen preprocessing pipeline then transforms the complete row "
            "before the accepted tuned XGBoost model produces the probability."
        )


def render_performance() -> None:
    st.subheader("Model Performance")
    metrics = load_metrics()

    cols = st.columns(5)
    values = [
        ("ROC-AUC", metrics["roc_auc"]),
        ("PR-AUC", metrics["pr_auc"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1", metrics["f1"]),
    ]
    for col, (label, value) in zip(cols, values):
        col.metric(label, f"{value:.4f}")

    st.markdown("### Final test confusion matrix")
    cm = pd.DataFrame(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ],
        index=["Actual Low Risk", "Actual High Risk"],
        columns=["Predicted Low Risk", "Predicted High Risk"],
    )
    st.dataframe(cm, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if ROC_FIGURE.exists():
            st.image(str(ROC_FIGURE), caption="Untouched test ROC curve")
        else:
            st.info("Run Slice 08 to generate the final ROC curve.")
    with c2:
        if PR_FIGURE.exists():
            st.image(str(PR_FIGURE), caption="Untouched test precision-recall curve")
        else:
            st.info("Run Slice 08 to generate the final PR curve.")

    st.caption(
        "Final test metrics were produced once after model and threshold "
        "selection. The test set was not used for tuning."
    )


def render_explainability() -> None:
    st.subheader("Why does the model make this decision?")
    if not IMPORTANCE_PATH.exists():
        st.warning(
            "Permutation importance has not been generated yet. Run "
            "python src\\explainability.py from the project root."
        )
        return

    importance = pd.read_csv(IMPORTANCE_PATH).sort_values(
        "importance_mean", ascending=False
    )
    top = importance.head(12).copy().sort_values("importance_mean")
    top["feature"] = (
        top["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    st.bar_chart(
        top.set_index("feature")["importance_mean"],
        horizontal=True,
    )
    st.caption(
        "Permutation importance is the mean drop in validation PR-AUC when "
        "a feature is randomly permuted. It is predictive signal, not causal evidence."
    )

    st.dataframe(
        importance.head(15)[
            ["rank", "feature", "importance_mean", "importance_std"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_methodology(artifacts: dict) -> None:
    st.subheader("Threshold & Model Selection")
    st.markdown(
        f"""
        **Accepted model:** {artifacts['model_name']}  
        **Frozen decision threshold:** {artifacts['threshold']:.2f}  
        **Selection metric:** validation PR-AUC, with ROC-AUC/F1 tie-breakers during tuning  
        **Threshold objective:** maximize validation F1  
        **Test set used for selection:** No
        """
    )

    comparison = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "Baseline XGBoost",
                "Tuned XGBoost",
            ],
            "Validation PR-AUC": [
                0.236606, 0.207645, 0.259062, 0.261614
            ],
            "Validation ROC-AUC": [
                0.759910, 0.740854, 0.772189, 0.774488
            ],
            "Validation F1": [
                0.269087, 0.277927, 0.291262, 0.305564
            ],
        }
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    threshold_data = pd.DataFrame(
        {
            "Operating point": ["Default 0.50", "Selected 0.58"],
            "Precision": [0.207335, 0.251090],
            "Recall": [0.580665, 0.475730],
            "F1": [0.305564, 0.328695],
        }
    )
    st.markdown("### Threshold decision")
    st.dataframe(threshold_data, use_container_width=True, hide_index=True)

    st.info(
        "The threshold is an operating decision selected on validation data "
        "to balance precision and recall through F1; it is not a probability "
        "calibration claim."
    )


def main() -> None:
    try:
        artifacts = load_artifacts()
    except Exception as exc:
        st.error(
            "The demo cannot start because required local ML artifacts are missing."
        )
        st.code(str(exc))
        st.markdown(
            "Run the pipeline through Slice 08 first, then launch with "
            "streamlit run app\\app.py."
        )
        st.stop()

    render_header(artifacts)

    st.sidebar.title("Demo Navigation")
    st.sidebar.caption("Final accepted pipeline")
    st.sidebar.code(
        f"{artifacts['model_name']}\\nthreshold = {artifacts['threshold']:.2f}"
    )

    page = st.sidebar.radio(
        "Open",
        [
            "Risk Assessment",
            "Model Performance",
            "Why This Prediction?",
            "Threshold & Selection",
        ],
    )

    if page == "Risk Assessment":
        render_predictor(artifacts)
    elif page == "Model Performance":
        render_performance()
    elif page == "Why This Prediction?":
        render_explainability()
    else:
        render_methodology(artifacts)

    st.sidebar.divider()
    st.sidebar.caption(
        "Demo-only interface · inference uses the frozen model and threshold · "
        "no training or test-set tuning"
    )


if __name__ == "__main__":
    main()
