# 🧠 Neuro-Forget  
### Explainable Alzheimer’s Detection with Machine Unlearning

---

## 🚀 Overview

**Neuro-Forget** is an end-to-end AI system for detecting Alzheimer’s disease from blood gene expression data.  
It combines **machine learning, explainable AI, and privacy-preserving techniques** into a single production-style pipeline.

---

## 🎯 Key Features

- 🧬 **Alzheimer’s classification** using transcriptomics data  
- 🤖 **XGBoost & LightGBM models**  
- 🔍 **Explainability with SHAP** (top gene importance)  
- 🧠 **Machine Unlearning (SISA)** – remove patient data efficiently  
- 🔐 **Privacy Audit** – membership inference attack  
- 🖥️ **Interactive UI (Streamlit)**  

---

## 📊 Dataset

- **GSE63060 (GEO)**
- Blood gene expression dataset for Alzheimer’s disease  

### Dataset Summary

| Metric | Value |
|------|------|
| Samples | 329 |
| Features | 136 |
| Classes | AD vs Control |

---

## 📈 Model Performance

| Model | Accuracy | F1 Score | ROC-AUC |
|------|--------|--------|--------|
| XGBoost | 0.712 | 0.667 | 0.797 |
| LightGBM | 0.712 | **0.678** | **0.804** |

✔ ROC-AUC ≈ **0.80** indicates strong predictive ability  

---

## 🔍 Explainability

SHAP analysis identifies key genes influencing predictions:

- ILMN_1776104  
- ILMN_1784286  
- ILMN_1749834  
- ILMN_2066060  

➡️ Provides **biological insight** instead of black-box predictions  

---

## 🧠 Machine Unlearning (SISA)

- Data is split into **shards**
- Each shard is trained independently
- When a patient is deleted:
  - Only affected shard is retrained  
  - No need for full retraining  

✔ Efficient and privacy-compliant  

---

## 🔐 Privacy Audit

- Membership inference attack simulation  
- Evaluates if the model memorizes training data  

✔ Helps assess **privacy risk in medical AI systems**

---

## 🖥️ Demo (Streamlit UI)

Run locally:

```bash
git clone https://github.com/Ahmedosamashouman/neuro-forget.git
cd neuro-forget

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

streamlit run app/Home.py
