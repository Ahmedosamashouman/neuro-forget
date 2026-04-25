from pathlib import Path
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Load data
X = pd.read_csv("data/gse63060/processed/X.csv", index_col=0)
y = pd.read_csv("data/gse63060/processed/y.csv", index_col=0)["label"]

# Clean feature names
X.columns = [
    re.sub(r"[^A-Za-z0-9_]", "_", str(col)).strip("_")
    for col in X.columns
]

# Model
model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("5-Fold CV ROC-AUC:", cv_scores.mean())

# Train/test evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = {
    "model": "XGBoost",
    "samples": len(X),
    "features": X.shape[1],
    "accuracy": accuracy_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, proba),
    "cv_roc_auc_mean": cv_scores.mean(),
    "cv_roc_auc_std": cv_scores.std(),
}

Path("models/gse63060_xgboost").mkdir(parents=True, exist_ok=True)
Path("reports/gse63060").mkdir(parents=True, exist_ok=True)

joblib.dump(model, "models/gse63060_xgboost/xgboost.joblib")
pd.DataFrame([metrics]).to_csv("reports/gse63060/xgboost_metrics.csv", index=False)

print("\n✅ XGBoost trained successfully")
print(metrics)