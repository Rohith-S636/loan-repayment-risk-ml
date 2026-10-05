"""
Loan Repayment Risk — Slice 09 polished Streamlit demo.

The app is inference-only. It never trains, tunes, or evaluates on the test
set. Prediction uses the frozen accepted model and threshold recorded by
Slice 07.
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
    .risk-card {
        padding: 28px; border-radius: 18px; text-align: center;
        background: #141d32; border: 1px solid #30405f;
    }
    .risk-prob { font-size: 3rem; font-weight: 800; margin: 8px 0; }
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
            <h1>Loan Repayment Risk</h1>
            <p>Explainable, imbalance-aware applicant risk assessment using
            tuned XGBoost · frozen threshold {artifacts['threshold']:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_predictor(artifacts: dict) -> None:
    st.subheader("Risk Predictor")
    st.caption(
        "Use a saved validation applicant for a reproducible demo, or explore "
        "a controlled scenario by changing key engineered risk indicators."
    )

    mode = st.radio(
        "Demo mode",
        ["Validation applicant", "Scenario explorer"],
        horizontal=True,
    )

    validation = artifacts["validation"]
    id_column = artifacts["id_column"]

    if mode == "Validation applicant":
        ids = validation[id_column].astype(int).tolist()
        selected_id = st.selectbox(
            "Applicant ID",
            ids,
            index=0,
            format_func=lambda x: f"SK_ID_CURR {x:,}",
        )
        row = validation[validation[id_column] == selected_id].copy()

        with st.expander("Applicant feature snapshot"):
            snapshot_cols = [
                c for c in [
                    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                    "APP_AGE_YEARS", "APP_EMPLOYED_YEARS",
                    "APP_CREDIT_TO_INCOME", "BUREAU_DEBT_TO_CREDIT",
                    "PREV_REFUSAL_RATIO", "INST_AVG_DELAY_POSITIVE",
                    "CC_HIGH_UTILIZATION_FLAG",
                ] if c in row.columns]
            st.dataframe(
                row[snapshot_cols].T.rename(columns={row.index[0]: "Value"}),
                use_container_width=True,
            )
    else:
        base = validation.iloc[[0]].copy()
        st.info(
            "Scenario explorer: changes are applied to engineered features "
            "while all other model features remain fixed from a validation-derived profile."
        )
        c1, c2 = st.columns(2)
        with c1:
            if "APP_AGE_YEARS" in base:
                base["APP_AGE_YEARS"] = st.slider(
                    "Age (years)", 18.0, 75.0,
                    float(np.clip(base["APP_AGE_YEARS"].iloc[0], 18, 75)),
                )
            if "APP_EMPLOYED_YEARS" in base:
                base["APP_EMPLOYED_YEARS"] = st.slider(
                    "Employment duration (years)", 0.0, 45.0,
                    float(np.clip(base["APP_EMPLOYED_YEARS"].iloc[0], 0, 45)),
                )
            if "APP_CREDIT_TO_INCOME" in base:
                base["APP_CREDIT_TO_INCOME"] = st.slider(
                    "Credit / income ratio", 0.1, 10.0,
                    float(np.clip(base["APP_CREDIT_TO_INCOME"].iloc[0], 0.1, 10)),
                )
            if "BUREAU_DEBT_TO_CREDIT" in base:
                base["BUREAU_DEBT_TO_CREDIT"] = st.slider(
                    "Bureau debt / credit", 0.0, 2.0,
                    float(np.clip(base["BUREAU_DEBT_TO_CREDIT"].iloc[0], 0, 2)),
                )
        with c2:
            if "APP_ANNUITY_TO_INCOME" in base:
                base["APP_ANNUITY_TO_INCOME"] = st.slider(
                    "Annuity / income ratio", 0.01, 1.0,
                    float(np.clip(base["APP_ANNUITY_TO_INCOME"].iloc[0], 0.01, 1)),
                )
            if "PREV_REFUSAL_RATIO" in base:
                base["PREV_REFUSAL_RATIO"] = st.slider(
                    "Previous refusal ratio", 0.0, 1.0,
                    float(np.clip(base["PREV_REFUSAL_RATIO"].iloc[0], 0, 1)),
                )
            if "INST_AVG_DELAY_POSITIVE" in base:
                base["INST_AVG_DELAY_POSITIVE"] = st.slider(
                    "Average installment delay (days)", 0.0, 60.0,
                    float(np.clip(base["INST_AVG_DELAY_POSITIVE"].iloc[0], 0, 60)),
                )
            if "CC_HIGH_UTILIZATION_FLAG" in base:
                base["CC_HIGH_UTILIZATION_FLAG"] = st.selectbox(
                    "High credit-card utilization",
                    [0, 1],
                    index=int(base["CC_HIGH_UTILIZATION_FLAG"].iloc[0]),
                    format_func=lambda x: "No" if x == 0 else "Yes",
                )
        row = base

    probability, prediction = get_prediction(artifacts, row)

    left, right = st.columns([1.1, 1])
    with left:
        risk_class = "risk-high" if prediction else "risk-low"
        label = "HIGH RISK" if prediction else "LOW RISK"
        st.markdown(
            f"""
            <div class="risk-card">
                <div class="small-note">Predicted probability of repayment difficulty</div>
                <div class="risk-prob {risk_class}">{probability:.1%}</div>
                <div class="{risk_class}" style="font-size:1.25rem;font-weight:700">{label}</div>
                <div class="small-note">Decision threshold: {artifacts['threshold']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.metric("Distance from threshold", f"{abs(probability - artifacts['threshold']):.3f}")
        st.metric("Model", artifacts["model_name"])
        st.caption(
            "The probability is produced by the frozen preprocessing pipeline "
            "and tuned XGBoost model. No training occurs in the app."
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
        st.markdown(
            "The explanation uses validation data only and reports the features "
            "whose permutation most reduces average precision."
        )
        return

    importance = pd.read_csv(IMPORTANCE_PATH).sort_values(
        "importance_mean", ascending=False
    )
    top = importance.head(12).copy().sort_values("importance_mean")
    top["feature"] = top["feature"].str.replace(
        "num__", "", regex=False
    ).str.replace("cat__", "", regex=False)

    st.bar_chart(
        top.set_index("feature")["importance_mean"],
        horizontal=True,
    )
    st.caption(
        "Permutation importance: mean drop in validation PR-AUC when a feature "
        "is randomly permuted. Positive values indicate useful predictive signal."
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
            "Model": ["Logistic Regression", "Random Forest", "Baseline XGBoost", "Tuned XGBoost"],
            "Validation PR-AUC": [0.236606, 0.207645, 0.259062, 0.261614],
            "Validation ROC-AUC": [0.759910, 0.740854, 0.772189, 0.774488],
            "Validation F1": [0.269087, 0.277927, 0.291262, 0.305564],
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
        "to balance precision and recall through F1; it is not a claim of "
        "probability calibration."
    )


def main() -> None:
    try:
        artifacts = load_artifacts()
    except Exception as exc:
        st.error("The demo cannot start because required local ML artifacts are missing.")
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
        ["Risk Predictor", "Model Performance", "Why This Prediction?", "Threshold & Selection"],
    )

    if page == "Risk Predictor":
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
