import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import joblib
import subprocess
from pathlib import Path
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.utils.cleaning import clean_feature_names


st.set_page_config(page_title="Neuro-Forget", layout="wide")

st.title("🧠 Neuro-Forget")
st.subheader("Explainable Alzheimer’s Detection with Machine Unlearning")

st.markdown("""
Neuro-Forget is an end-to-end bioinformatics AI system for Alzheimer’s disease detection.
It combines transcriptomics-based classification, SHAP explainability, SISA-style machine
unlearning, MIA privacy validation, and a FASTA/k-mer demo module.
""")

# =========================
# PATHS
# =========================
DATASET = "GSE63060"

X_PATH = Path("data/gse63060/processed/X.csv")
Y_PATH = Path("data/gse63060/processed/y.csv")

XGB_MODEL_PATH = Path("models/gse63060_xgboost/xgboost.joblib")
LGBM_MODEL_PATH = Path("models/gse63060_lightgbm/lightgbm.joblib")

XGB_METRICS_PATH = Path("reports/gse63060/xgboost_metrics.csv")
LGBM_METRICS_PATH = Path("reports/gse63060/lightgbm_metrics.csv")

SHAP_PATH = Path("reports/figures/shap_summary.png")
SHAP_TABLE = Path("reports/tables/top_shap_genes.csv")

PRIVACY_AUDIT_PATH = Path("reports/tables/privacy_audit.csv")
MIA_METRICS_PATH = Path("reports/tables/mia_metrics.csv")
MIA_BEFORE_AFTER_PATH = Path("reports/tables/mia_before_after_metrics.csv")
DELETED_SAMPLE_VERIFICATION_PATH = Path("reports/tables/deleted_sample_verification.csv")

DELETION_LOG_PATH = Path("reports/tables/deletion_log.csv")
UNLEARNING_BENCHMARK_PATH = Path("reports/tables/unlearning_vs_retraining.csv")

FASTA_X_PATH = Path("data/fasta_demo/X_kmers.csv")
FASTA_Y_PATH = Path("data/fasta_demo/y_kmers.csv")
FASTA_PATH = Path("data/fasta_demo/alzheimers_simulated.fasta")

# =========================
# LOAD MAIN DATA
# =========================
X = pd.read_csv(X_PATH, index_col=0)
X = clean_feature_names(X)
y = pd.read_csv(Y_PATH, index_col=0)["label"]
y.index = X.index

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")

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

st.sidebar.info(f"Current model: {model_choice}")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Dataset",
        "Model Metrics",
        "Predict",
        "Explainability",
        "Machine Unlearning",
        "Privacy Audit",
        "FASTA / k-mer Demo",
    ]
)

# =========================
# OVERVIEW
# =========================
if page == "Overview":
    st.header("Project Overview")

    st.write("""
    This project detects Alzheimer’s disease using blood transcriptomics data and adds
    privacy-preserving functionality through machine unlearning and membership inference
    attack validation.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Main Dataset", DATASET)
    col2.metric("Samples", X.shape[0])
    col3.metric("Gene Features", X.shape[1])

    st.subheader("Main Contributions")
    st.markdown("""
    - Alzheimer’s disease classification using **XGBoost** and **LightGBM**
    - Explainability using **SHAP**
    - Patient-level deletion using **SISA-style unlearning**
    - Privacy validation using **Membership Inference Attack**
    - FASTA-to-k-mer demo module for sequence-processing requirement
    - Interactive **Streamlit** interface
    """)

# =========================
# DATASET
# =========================
elif page == "Dataset":
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset", DATASET)
    col2.metric("Samples", X.shape[0])
    col3.metric("Selected Features", X.shape[1])

    st.write("Labels: **0 = Control**, **1 = Alzheimer’s Disease**")
    st.subheader("Class Distribution")
    st.dataframe(y.value_counts().rename("count"))

    st.subheader("Dataset Preview")
    st.dataframe(X.head())

# =========================
# MODEL METRICS
# =========================
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
            col4, col5 = st.columns(2)
            col4.metric("CV ROC-AUC Mean", round(float(metrics["cv_roc_auc_mean"].iloc[0]), 3))
            col5.metric("CV ROC-AUC Std", round(float(metrics["cv_roc_auc_std"].iloc[0]), 3))
    else:
        st.warning("Metrics file not found. Train the selected model first.")

# =========================
# PREDICT
# =========================
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

# =========================
# EXPLAINABILITY
# =========================
elif page == "Explainability":
    st.header("SHAP Explainability")

    st.write("""
    SHAP explains how much each gene feature contributes to the model prediction.
    Positive SHAP values push the prediction toward Alzheimer’s disease, while negative
    values push toward Control.
    """)

    if SHAP_PATH.exists():
        st.image(str(SHAP_PATH), caption="SHAP Summary Plot")
    else:
        st.warning("SHAP image not found. Run explain_shap.py first.")

    if SHAP_TABLE.exists():
        st.subheader("Top Important Gene Probes")
        st.dataframe(pd.read_csv(SHAP_TABLE).head(25))
    else:
        st.warning("SHAP table not found.")

# =========================
# MACHINE UNLEARNING
# =========================
elif page == "Machine Unlearning":
    st.header("Machine Unlearning: SISA-style Patient Deletion")

    st.write("""
    SISA means **Sharded, Isolated, Sliced, and Aggregated** training.
    Instead of retraining the whole model after deleting one patient, the system retrains
    only the affected shard.
    """)

    sample_id = st.selectbox("Choose patient/sample to forget", X.index.tolist())

    if st.button("Forget this patient"):
        result = subprocess.run(
            [sys.executable, "-m", "src.unlearning.delete_patient", sample_id],
            capture_output=True,
            text=True
        )

        st.subheader("Deletion Output")
        st.code(result.stdout)

        if result.stderr:
            st.error(result.stderr)

    if DELETION_LOG_PATH.exists():
        st.subheader("Deletion Log")
        st.dataframe(pd.read_csv(DELETION_LOG_PATH))

    if UNLEARNING_BENCHMARK_PATH.exists():
        st.subheader("Unlearning vs Full Retraining")
        benchmark = pd.read_csv(UNLEARNING_BENCHMARK_PATH)
        st.dataframe(benchmark)

        if "time_sec" in benchmark.columns:
            st.bar_chart(benchmark.set_index("method")["time_sec"])

    if DELETED_SAMPLE_VERIFICATION_PATH.exists():
        st.subheader("Deleted Sample Verification")
        st.dataframe(pd.read_csv(DELETED_SAMPLE_VERIFICATION_PATH))
    else:
        st.warning("Deleted sample verification not found. Run benchmark_unlearning_vs_retraining.py first.")

# =========================
# PRIVACY AUDIT
# =========================
elif page == "Privacy Audit":
    st.header("Privacy Audit: Membership Inference Attack")

    st.write("""
    A Membership Inference Attack tries to determine whether a specific patient sample
    was used during model training. Higher attack accuracy means greater privacy risk.
    """)

    if PRIVACY_AUDIT_PATH.exists():
        st.subheader("Confidence / Entropy Privacy Audit")
        audit = pd.read_csv(PRIVACY_AUDIT_PATH)
        st.dataframe(audit)

        if "sample_id" in audit.columns and "confidence" in audit.columns:
            st.subheader("Confidence Chart")
            st.bar_chart(audit.set_index("sample_id")["confidence"])
    else:
        st.warning("Privacy audit file not found. Run membership_inference.py first.")

    if MIA_BEFORE_AFTER_PATH.exists():
        st.subheader("MIA Before vs After Unlearning")
        mia_metrics = pd.read_csv(MIA_BEFORE_AFTER_PATH)
        st.dataframe(mia_metrics)

        before = mia_metrics[mia_metrics["stage"] == "Before unlearning"].iloc[0]
        after = mia_metrics.iloc[1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Before MIA Accuracy", round(float(before["mia_accuracy"]), 3))
        col2.metric("After MIA Accuracy", round(float(after["mia_accuracy"]), 3))
        col3.metric("After MIA AUC", round(float(after["mia_auc"]), 3))

    elif MIA_METRICS_PATH.exists():
        st.subheader("MIA Attack Metrics")
        mia_metrics = pd.read_csv(MIA_METRICS_PATH)
        st.dataframe(mia_metrics)

        col1, col2, col3 = st.columns(3)
        col1.metric("MIA Accuracy", round(float(mia_metrics["mia_accuracy"].iloc[0]), 3))
        col2.metric("MIA F1", round(float(mia_metrics["mia_f1"].iloc[0]), 3))
        col3.metric("MIA AUC", round(float(mia_metrics["mia_auc"].iloc[0]), 3))
    else:
        st.warning("MIA metrics not found. Run mia_before_after_unlearning.py first.")

# =========================
# FASTA / K-MER DEMO
# =========================
elif page == "FASTA / k-mer Demo":
    st.header("FASTA / k-mer Demo Module")

    st.write("""
    This module demonstrates the official project requirement of converting raw DNA-like
    FASTA sequences into numerical features using k-mer counting.
    """)

    if FASTA_X_PATH.exists() and FASTA_Y_PATH.exists():
        X_kmer = pd.read_csv(FASTA_X_PATH, index_col=0)
        y_kmer = pd.read_csv(FASTA_Y_PATH, index_col=0)["label"]

        col1, col2, col3 = st.columns(3)
        col1.metric("FASTA File", "Generated")
        col2.metric("Samples", X_kmer.shape[0])
        col3.metric("k-mer Features", X_kmer.shape[1])

        st.subheader("FASTA Label Distribution")
        st.dataframe(y_kmer.value_counts().rename("count"))

        st.subheader("k-mer Feature Preview")
        st.dataframe(X_kmer.head())

        if FASTA_PATH.exists():
            st.subheader("Example FASTA Records")
            with open(FASTA_PATH, "r") as f:
                lines = f.readlines()[:8]
            st.code("".join(lines))
    else:
        st.warning("FASTA demo files not found. Run fasta_kmer_demo.py first.")
