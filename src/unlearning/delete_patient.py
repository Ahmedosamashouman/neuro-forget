import sys
import time
from pathlib import Path
import pandas as pd
import joblib
from xgboost import XGBClassifier

from src.utils.cleaning import clean_feature_names

if len(sys.argv) < 2:
    print("Usage: python -m src.unlearning.delete_patient SAMPLE_ID")
    sys.exit(1)

sample_id = sys.argv[1].replace('"', '').replace("'", "")

X = pd.read_csv("data/gse63060/processed/X.csv", index_col=0)
y = pd.read_csv("data/gse63060/processed/y.csv", index_col=0)["label"]

X = clean_feature_names(X)
y.index = X.index

REPORT = Path("reports/tables")
REPORT.mkdir(parents=True, exist_ok=True)

mapping_path = REPORT / "patient_to_shard.csv"

if not mapping_path.exists():
    print("patient_to_shard.csv not found. Run:")
    print("python -m src.unlearning.benchmark_unlearning_vs_retraining")
    sys.exit(1)

mapping = pd.read_csv(mapping_path)
mapping["sample_id"] = mapping["sample_id"].astype(str).str.replace('"', '').str.replace("'", "")

if sample_id not in mapping["sample_id"].values:
    print(f"Sample not found: {sample_id}")
    sys.exit(1)

affected_shard = int(mapping.loc[mapping["sample_id"] == sample_id, "shard"].iloc[0])

shard_samples = mapping[mapping["shard"] == affected_shard]["sample_id"].tolist()
remaining = [s for s in shard_samples if s != sample_id and s in X.index]

start = time.time()

model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X.loc[remaining], y.loc[remaining])

Path("models/shards").mkdir(parents=True, exist_ok=True)
joblib.dump(model, f"models/shards/shard_{affected_shard}.joblib")

elapsed = time.time() - start

log = pd.DataFrame([{
    "deleted_sample": sample_id,
    "affected_shard": affected_shard,
    "remaining_samples_in_shard": len(remaining),
    "unlearning_time_sec": elapsed
}])

log.to_csv(REPORT / "deletion_log.csv", index=False)

print("Patient forgotten successfully")
print(log)
