from pathlib import Path
import pandas as pd
import numpy as np
import gzip
from sklearn.preprocessing import StandardScaler

RAW = Path("data/gse63060/raw")
INTERIM = Path("data/gse63060/interim")
PROCESSED = Path("data/gse63060/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

FILE = RAW / "GSE63060_series_matrix.txt.gz"

meta = pd.read_csv(INTERIM / "metadata.csv")

data = []
genes = []

with gzip.open(FILE, "rt", errors="ignore") as f:
    reading = False
    for line in f:
        if line.startswith("!series_matrix_table_begin"):
            reading = True
            continue
        if line.startswith("!series_matrix_table_end"):
            break
        if reading:
            parts = line.strip().split("\t")
            genes.append(parts[0])
            data.append(parts[1:])

X = pd.DataFrame(data).T
X.columns = genes
X.index = meta["sample_id"].values[:len(X)]

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

X = np.log2(X + 1)

# feature selection
X = X.loc[:, X.var() > 0.01]
top_genes = X.var().sort_values(ascending=False).head(1000).index
X = X[top_genes]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

y = meta.set_index("sample_id").loc[X_scaled.index, "label"]
y = y.map({"CTRL": 0, "AD": 1})

X_scaled.to_csv(PROCESSED / "X.csv")
y.to_csv(PROCESSED / "y.csv")

print("Dataset ready:", X_scaled.shape)
print(y.value_counts())
