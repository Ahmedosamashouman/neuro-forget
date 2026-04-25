from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROCESSED = Path("data/processed")
MODEL_DIR = Path("models/baseline")
REPORTS = Path("reports/tables")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

X = pd.read_csv(PROCESSED / "X_ad_ctrl.csv", index_col=0)
y = pd.read_csv(PROCESSED / "y_ad_ctrl.csv", index_col=0)["label"]

model = LogisticRegression(max_iter=2000, class_weight="balanced")

loo = LeaveOneOut()
proba = cross_val_predict(model, X, y, cv=loo, method="predict_proba")[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = {
    "model": "LogisticRegression",
    "accuracy": accuracy_score(y, pred),
    "f1": f1_score(y, pred),
    "roc_auc": roc_auc_score(y, proba),
}

pd.DataFrame([metrics]).to_csv(REPORTS / "baseline_metrics.csv", index=False)

model.fit(X, y)
joblib.dump(model, MODEL_DIR / "logreg.joblib")

print(metrics)