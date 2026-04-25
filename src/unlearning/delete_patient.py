from pathlib import Path
import time
import sys
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

sample_to_delete = sys.argv[1] if len(sys.argv) > 1 else None
if sample_to_delete is None:
    raise SystemExit("Usage: python src/unlearning/delete_patient.py GSM_ID")

X = pd.read_csv("data/processed/X_ad_ctrl.csv", index_col=0)
y = pd.read_csv("data/processed/y_ad_ctrl.csv", index_col=0)["label"]
mapping = pd.read_csv("reports/tables/patient_to_shard.csv")

row = mapping[mapping["sample_id"] == sample_to_delete]
if row.empty:
    raise SystemExit(f"Sample not found: {sample_to_delete}")

shard = int(row.iloc[0]["shard"])
ids = mapping[(mapping["shard"] == shard) & (mapping["sample_id"] != sample_to_delete)]["sample_id"]

start = time.time()
model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X.loc[ids], y.loc[ids])
unlearn_time = time.time() - start

joblib.dump(model, f"models/shards/shard_{shard}.joblib")

log = pd.DataFrame([{
    "deleted_sample": sample_to_delete,
    "affected_shard": shard,
    "remaining_samples_in_shard": len(ids),
    "unlearning_time_sec": unlearn_time
}])
log.to_csv("reports/tables/deletion_log.csv", index=False)

print("Patient forgotten from affected shard.")
print(log)
