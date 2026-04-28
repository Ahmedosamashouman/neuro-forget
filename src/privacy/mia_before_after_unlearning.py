from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.utils.cleaning import clean_feature_names

REPORT = Path("reports/tables")
REPORT.mkdir(parents=True, exist_ok=True)

X = pd.read_csv("data/gse63060/processed/X.csv", index_col=0)
y = pd.read_csv("data/gse63060/processed/y.csv", index_col=0)["label"]

X = clean_feature_names(X)
y.index = X.index

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    stratify=y,
    random_state=42
)

# Delete more than one sample to make privacy effect clearer
N_DELETE = 10
deleted_samples = list(X_train.index[:N_DELETE])

def train_target_model(Xtr, ytr):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(Xtr, ytr)
    return model

def build_attack_features(model, X_data):
    probs = model.predict_proba(X_data)
    probability_ad = probs[:, 1]
    confidence = probs.max(axis=1)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)

    return pd.DataFrame({
        "probability_AD": probability_ad,
        "confidence": confidence,
        "entropy": entropy
    })

def run_mia(model, X_member, X_nonmember, stage):
    member_df = build_attack_features(model, X_member)
    member_df["is_member"] = 1

    nonmember_df = build_attack_features(model, X_nonmember)
    nonmember_df["is_member"] = 0

    attack_df = pd.concat([member_df, nonmember_df], axis=0).reset_index(drop=True)

    Xa = attack_df[["probability_AD", "confidence", "entropy"]]
    ya = attack_df["is_member"]

    Xa_train, Xa_test, ya_train, ya_test = train_test_split(
        Xa,
        ya,
        test_size=0.3,
        stratify=ya,
        random_state=42
    )

    attack_model = LogisticRegression(max_iter=1000)
    attack_model.fit(Xa_train, ya_train)

    pred = attack_model.predict(Xa_test)
    proba = attack_model.predict_proba(Xa_test)[:, 1]

    return {
        "stage": stage,
        "deleted_samples_count": len(deleted_samples),
        "mia_accuracy": accuracy_score(ya_test, pred),
        "mia_f1": f1_score(ya_test, pred),
        "mia_auc": roc_auc_score(ya_test, proba),
        "meaning": "Lower MIA accuracy after deletion means lower membership leakage"
    }

# BEFORE unlearning
model_before = train_target_model(X_train, y_train)

mia_before = run_mia(
    model_before,
    X_train,
    X_test,
    "Before unlearning"
)

# AFTER unlearning
X_train_after = X_train.drop(index=deleted_samples, errors="ignore")
y_train_after = y_train.drop(index=deleted_samples, errors="ignore")

model_after = train_target_model(X_train_after, y_train_after)

mia_after = run_mia(
    model_after,
    X_train_after,
    X_test,
    "After deleting patients"
)

# Save MIA before/after
results = pd.DataFrame([mia_before, mia_after])
results.to_csv(REPORT / "mia_before_after_metrics.csv", index=False)

# Per-sample forgetting proof
before_probs = model_before.predict_proba(X.loc[deleted_samples])
after_probs = model_after.predict_proba(X.loc[deleted_samples])

forgetting_rows = []

for i, sample_id in enumerate(deleted_samples):
    before_conf = float(before_probs[i].max())
    after_conf = float(after_probs[i].max())

    before_entropy = float(-np.sum(before_probs[i] * np.log(before_probs[i] + 1e-12)))
    after_entropy = float(-np.sum(after_probs[i] * np.log(after_probs[i] + 1e-12)))

    forgetting_rows.append({
        "deleted_sample": sample_id,
        "true_label": int(y.loc[sample_id]),
        "confidence_before": before_conf,
        "confidence_after": after_conf,
        "confidence_change": after_conf - before_conf,
        "entropy_before": before_entropy,
        "entropy_after": after_entropy,
        "entropy_change": after_entropy - before_entropy
    })

forgetting_df = pd.DataFrame(forgetting_rows)
forgetting_df.to_csv(REPORT / "per_sample_forgetting_proof.csv", index=False)

print("MIA before/after:")
print(results)

print("\nPer-sample forgetting proof:")
print(forgetting_df)

print("\nSaved:")
print(REPORT / "mia_before_after_metrics.csv")
print(REPORT / "per_sample_forgetting_proof.csv")
