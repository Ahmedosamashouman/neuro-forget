from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier

from src.utils.cleaning import clean_feature_names

X = pd.read_csv("data/gse63060/processed/X.csv", index_col=0)
y = pd.read_csv("data/gse63060/processed/y.csv", index_col=0)["label"]

X = clean_feature_names(X)
y.index = X.index

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    verbose=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = {
    "model": "LightGBM",
    "samples": len(X),
    "features": X.shape[1],
    "accuracy": accuracy_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, proba),
    "cv_roc_auc_mean": cv_scores.mean(),
    "cv_roc_auc_std": cv_scores.std(),
}

Path("models/gse63060_lightgbm").mkdir(parents=True, exist_ok=True)
Path("reports/gse63060").mkdir(parents=True, exist_ok=True)

joblib.dump(model, "models/gse63060_lightgbm/lightgbm.joblib")
pd.DataFrame([metrics]).to_csv("reports/gse63060/lightgbm_metrics.csv", index=False)

print(metrics)
