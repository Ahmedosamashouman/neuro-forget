from pathlib import Path
import pandas as pd
import numpy as np
import joblib

Path("reports/tables").mkdir(parents=True, exist_ok=True)

X = pd.read_csv("data/processed/X_ad_ctrl.csv", index_col=0)
y = pd.read_csv("data/processed/y_ad_ctrl.csv", index_col=0)["label"]

model = joblib.load("models/xgboost/xgboost.joblib")
proba = model.predict_proba(X)[:, 1]

audit = pd.DataFrame({
    "sample_id": X.index,
    "true_label": y.values,
    "probability_AD": proba,
    "confidence": np.maximum(proba, 1 - proba),
    "entropy": -(proba*np.log(proba + 1e-9) + (1-proba)*np.log(1-proba + 1e-9))
})

audit.to_csv("reports/tables/privacy_audit.csv", index=False)

print("Privacy audit saved: reports/tables/privacy_audit.csv")
print(audit.head())
