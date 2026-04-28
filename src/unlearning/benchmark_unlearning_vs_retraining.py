from pathlib import Path
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.utils.cleaning import clean_feature_names

X = pd.read_csv("data/gse63060/processed/X.csv", index_col=0)
y = pd.read_csv("data/gse63060/processed/y.csv", index_col=0)["label"]

X = clean_feature_names(X)
y.index = X.index

REPORT = Path("reports/tables")
REPORT.mkdir(parents=True, exist_ok=True)

delete_sample = X.index[0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

def train_model(Xtr, ytr):
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

def evaluate_ensemble(models, X_eval, y_eval):
    probs = np.mean([m.predict_proba(X_eval)[:, 1] for m in models], axis=0)
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_eval, preds),
        "f1": f1_score(y_eval, preds),
        "roc_auc": roc_auc_score(y_eval, probs)
    }

# original SISA ensemble
n_shards = 3
train_ids = list(X_train.index)
shards = [train_ids[i::n_shards] for i in range(n_shards)]

original_models = []
patient_to_shard = []

for shard_id, ids in enumerate(shards):
    Xs = X_train.loc[ids]
    ys = y_train.loc[ids]

    for sid in ids:
        patient_to_shard.append({"sample_id": sid, "shard": shard_id})

    model = train_model(Xs, ys)
    original_models.append(model)

original_metrics = evaluate_ensemble(original_models, X_test, y_test)

patient_to_shard = pd.DataFrame(patient_to_shard)
patient_to_shard.to_csv(REPORT / "patient_to_shard.csv", index=False)

if delete_sample not in patient_to_shard["sample_id"].values:
    delete_sample = patient_to_shard["sample_id"].iloc[0]

affected_shard = int(patient_to_shard.loc[
    patient_to_shard["sample_id"] == delete_sample, "shard"
].iloc[0])

# prediction before deletion
before_prob = float(np.mean([
    m.predict_proba(X.loc[[delete_sample]])[:, 1][0]
    for m in original_models
]))

# SISA unlearning: retrain only affected shard
sisa_start = time.time()

sisa_models = original_models.copy()
affected_ids = patient_to_shard[
    patient_to_shard["shard"] == affected_shard
]["sample_id"].tolist()

affected_ids = [sid for sid in affected_ids if sid != delete_sample]

X_affected = X.loc[affected_ids]
y_affected = y.loc[affected_ids]

sisa_models[affected_shard] = train_model(X_affected, y_affected)

sisa_time = time.time() - sisa_start
sisa_metrics = evaluate_ensemble(sisa_models, X_test, y_test)

after_prob = float(np.mean([
    m.predict_proba(X.loc[[delete_sample]])[:, 1][0]
    for m in sisa_models
]))

# full retraining after deleting patient
full_start = time.time()

X_train_full = X_train.drop(index=delete_sample, errors="ignore")
y_train_full = y_train.drop(index=delete_sample, errors="ignore")

full_model = train_model(X_train_full, y_train_full)
full_time = time.time() - full_start

full_proba = full_model.predict_proba(X_test)[:, 1]
full_pred = (full_proba >= 0.5).astype(int)

full_metrics = {
    "accuracy": accuracy_score(y_test, full_pred),
    "f1": f1_score(y_test, full_pred),
    "roc_auc": roc_auc_score(y_test, full_proba)
}

results = pd.DataFrame([
    {
        "method": "Original SISA ensemble",
        "deleted_sample": delete_sample,
        "affected_shard": affected_shard,
        "time_sec": 0,
        **original_metrics
    },
    {
        "method": "SISA unlearning",
        "deleted_sample": delete_sample,
        "affected_shard": affected_shard,
        "time_sec": sisa_time,
        **sisa_metrics
    },
    {
        "method": "Full retraining",
        "deleted_sample": delete_sample,
        "affected_shard": "all",
        "time_sec": full_time,
        **full_metrics
    }
])

results["speedup_vs_full_retraining"] = full_time / results["time_sec"].replace(0, np.nan)

verification = pd.DataFrame([{
    "deleted_sample": delete_sample,
    "affected_shard": affected_shard,
    "probability_AD_before_unlearning": before_prob,
    "probability_AD_after_unlearning": after_prob,
    "absolute_probability_change": abs(before_prob - after_prob)
}])

results.to_csv(REPORT / "unlearning_vs_retraining.csv", index=False)
verification.to_csv(REPORT / "deleted_sample_verification.csv", index=False)

print(results)
print(verification)
