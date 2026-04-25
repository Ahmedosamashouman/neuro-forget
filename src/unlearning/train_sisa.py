from pathlib import Path
import time
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

X = pd.read_csv("data/processed/X_ad_ctrl.csv", index_col=0)
y = pd.read_csv("data/processed/y_ad_ctrl.csv", index_col=0)["label"]

Path("models/shards").mkdir(parents=True, exist_ok=True)
Path("reports/tables").mkdir(parents=True, exist_ok=True)

n_shards = 3
mapping = pd.DataFrame({
    "sample_id": X.index,
    "shard": [i % n_shards for i in range(len(X))]
})
mapping.to_csv("reports/tables/patient_to_shard.csv", index=False)

rows = []
for shard in range(n_shards):
    ids = mapping[mapping["shard"] == shard]["sample_id"]
    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    start = time.time()
    model.fit(X.loc[ids], y.loc[ids])
    train_time = time.time() - start

    joblib.dump(model, f"models/shards/shard_{shard}.joblib")
    rows.append({"shard": shard, "samples": len(ids), "train_time_sec": train_time})

pd.DataFrame(rows).to_csv("reports/tables/sisa_train_times.csv", index=False)

print("SISA shard models saved in models/shards/")
print(pd.DataFrame(rows))
