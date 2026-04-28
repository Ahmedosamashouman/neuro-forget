from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

X = pd.read_csv("data/gse63060/processed/X.csv", index_col=0)
y = pd.read_csv("data/gse63060/processed/y.csv", index_col=0)["label"]

REPORT = Path("reports/tables")
REPORT.mkdir(parents=True, exist_ok=True)

# Split into target-model train/test
X_train, X_nonmember, y_train, y_nonmember = train_test_split(
    X, y,
    test_size=0.4,
    stratify=y,
    random_state=42
)

target_model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

target_model.fit(X_train, y_train)

# Members = samples used in training
member_probs = target_model.predict_proba(X_train)
nonmember_probs = target_model.predict_proba(X_nonmember)

def build_attack_features(probs):
    confidence = probs.max(axis=1)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    prob_ad = probs[:, 1]
    return pd.DataFrame({
        "confidence": confidence,
        "entropy": entropy,
        "probability_AD": prob_ad
    })

member_features = build_attack_features(member_probs)
member_features["is_member"] = 1

nonmember_features = build_attack_features(nonmember_probs)
nonmember_features["is_member"] = 0

attack_data = pd.concat([member_features, nonmember_features], axis=0).reset_index(drop=True)

Xa = attack_data[["confidence", "entropy", "probability_AD"]]
ya = attack_data["is_member"]

Xa_train, Xa_test, ya_train, ya_test = train_test_split(
    Xa, ya,
    test_size=0.3,
    stratify=ya,
    random_state=42
)

attack_model = LogisticRegression(max_iter=1000)
attack_model.fit(Xa_train, ya_train)

attack_proba = attack_model.predict_proba(Xa_test)[:, 1]
attack_pred = (attack_proba >= 0.5).astype(int)

metrics = {
    "mia_accuracy": accuracy_score(ya_test, attack_pred),
    "mia_f1": f1_score(ya_test, attack_pred),
    "mia_auc": roc_auc_score(ya_test, attack_proba),
    "attack_features": "confidence, entropy, probability_AD",
    "meaning": "Predicts whether a sample was used in target model training"
}

pd.DataFrame([metrics]).to_csv(REPORT / "mia_metrics.csv", index=False)
attack_data.to_csv(REPORT / "mia_attack_dataset.csv", index=False)

print(metrics)
print("Saved:", REPORT / "mia_metrics.csv")
