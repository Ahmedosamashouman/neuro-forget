from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

PROCESSED = Path("data/processed")
MODEL_DIR = Path("models/xgboost")
REPORTS = Path("reports/tables")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

X = pd.read_csv(PROCESSED / "X_ad_ctrl.csv", index_col=0)
y = pd.read_csv(PROCESSED / "y_ad_ctrl.csv", index_col=0)["label"]

model = XGBClassifier(
    n_estimators=80,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

loo = LeaveOneOut()
proba = cross_val_predict(model, X, y, cv=loo, method="predict_proba")[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = {
    "model": "XGBoost",
    "accuracy": accuracy_score(y, pred),
    "f1": f1_score(y, pred),
    "roc_auc": roc_auc_score(y, proba),
}

pd.DataFrame([metrics]).to_csv(REPORTS / "xgboost_metrics.csv", index=False)

model.fit(X, y)
joblib.dump(model, MODEL_DIR / "xgboost.joblib")

print(metrics)
print("Saved model to models/xgboost/xgboost.joblib")
