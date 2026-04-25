import streamlit as st
import pandas as pd
import joblib
import subprocess
from pathlib import Path

st.set_page_config(page_title="Neuro-Forget", layout="wide")

st.title("🧠 Neuro-Forget")
st.subheader("Explainable Alzheimer’s detection with patient-level machine unlearning")

st.markdown("""
This demo uses public Alzheimer’s blood transcriptomics data from **GSE63060**
to classify Alzheimer’s disease vs control, explain predictions using SHAP,
and demonstrate patient-level SISA-style machine unlearning.
""")

DATASET = "GSE63060"

X_PATH = Path("data/gse63060/processed/X.csv")
Y_PATH = Path("data/gse63060/processed/y.csv")

XGB_MODEL_PATH = Path("models/gse63060_xgboost/xgboost.joblib")
LGBM_MODEL_PATH = Path("models/gse63060_lightgbm/lightgbm.joblib")

XGB_METRICS_PATH = Path("reports/gse63060/xgboost_metrics.csv")
LGBM_METRICS_PATH = Path("reports/gse63060/lightgbm_metrics.csv")

SHAP_PATH = Path("reports/figures/shap_summary.png")
SHAP_TABLE = Path("reports/tables/top_shap_genes.csv")

X = pd.read_csv(X_PATH, index_col=0)
y = pd.read_csv(Y_PATH, index_col=0)["label"]

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["XGBoost", "LightGBM"]
)

if model_choice == "XGBoost":
    model = joblib.load(XGB_MODEL_PATH)
    metrics_path = XGB_METRICS_PATH
else:
    model = joblib.load(LGBM_MODEL_PATH)
    metrics_path = LGBM_METRICS_PATH

st.sidebar.markdown("---")
st.sidebar.info(f"Current model: {model_choice}")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Dataset",
        "Model Metrics",
        "Predict",
        "Explainability",
        "Delete Patient",
        "Privacy Audit",
    ]
)

if page == "Overview":
    st.header("Project Overview")
    st.write("""
    **Neuro-Forget** is an end-to-end Alzheimer’s disease detection system using
    blood transcriptomics. It combines machine learning, explainable AI,
    patient-level machine unlearning, and privacy auditing.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Main dataset", DATASET)
    col2.metric("Samples", X.shape[0])
    col3.metric("Gene features", X.shape[1])

    st.success("Main dataset: GSE63060")
    st.info("Prototype dataset: GSE161199")

elif page == "Dataset":
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset", DATASET)
    col2.metric("Samples", X.shape[0])
    col3.metric("Selected gene features", X.shape[1])

    st.write("Labels: 0 = Control, 1 = Alzheimer’s Disease")
    st.dataframe(y.value_counts().rename("count"))

    st.subheader("Preview")
    st.dataframe(X.head())

elif page == "Model Metrics":
    st.header("Model Performance")
    st.info(f"Showing metrics for: {model_choice}")

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics)

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(float(metrics["accuracy"].iloc[0]), 3))
        col2.metric("F1-score", round(float(metrics["f1"].iloc[0]), 3))
        col3.metric("ROC-AUC", round(float(metrics["roc_auc"].iloc[0]), 3))

        if "cv_roc_auc_mean" in metrics.columns:
            st.metric(
                "5-Fold CV ROC-AUC",
                round(float(metrics["cv_roc_auc_mean"].iloc[0]), 3)
            )
    else:
        st.warning("Metrics file not found. Train the selected model first.")

elif page == "Predict":
    st.header("Predict Alzheimer’s Disease")
    st.info(f"Using model: {model_choice}")

    sample_id = st.selectbox("Choose sample", X.index.tolist())

    if st.button("Predict"):
        proba = float(model.predict_proba(X.loc[[sample_id]])[0, 1])
        pred = "Alzheimer’s Disease" if proba >= 0.5 else "Control"

        col1, col2 = st.columns(2)
        col1.metric("Prediction", pred)
        col2.metric("Probability of AD", round(proba, 4))

elif page == "Explainability":
    st.header("SHAP Explainability")

    if SHAP_PATH.exists():
        st.image(str(SHAP_PATH), caption="SHAP summary plot")
    else:
        st.warning("SHAP image not found. Run explain_shap.py first.")

    if SHAP_TABLE.exists():
        st.subheader("Top important gene probes")
        st.dataframe(pd.read_csv(SHAP_TABLE).head(20))
    else:
        st.warning("SHAP table not found.")

elif page == "Delete Patient":
    st.header("Machine Unlearning")
    sample_id = st.selectbox("Choose patient/sample to forget", X.index.tolist())

    st.write("""
    This demo removes a sample from its assigned shard and retrains only the affected shard.
    This simulates patient-level machine unlearning.
    """)

    if st.button("Forget this patient"):
        result = subprocess.run(
            ["python", "src/unlearning/delete_patient.py", sample_id],
            capture_output=True,
            text=True
        )

        st.code(result.stdout)

        if Path("reports/tables/deletion_log.csv").exists():
            st.dataframe(pd.read_csv("reports/tables/deletion_log.csv"))

elif page == "Privacy Audit":
    st.header("Membership Inference Privacy Audit")

    audit_path = Path("reports/tables/privacy_audit.csv")

    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        st.write("Higher confidence can indicate higher memorization risk.")
        st.dataframe(audit)

        if "sample_id" in audit.columns and "confidence" in audit.columns:
            st.bar_chart(audit.set_index("sample_id")["confidence"])
    else:
        st.warning("Privacy audit file not found. Run membership_inference.py first.")