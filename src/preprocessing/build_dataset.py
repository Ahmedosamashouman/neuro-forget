from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

RAW = Path("data/raw")
INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

EXPR = RAW / "GSE161199_norm_counts_FPKM_GRCh38.p13_NCBI.tsv.gz"

# load metadata
meta = pd.read_csv(INTERIM / "metadata.csv")
meta = meta[meta["label"].isin(["AD", "CTRL"])].copy()

print("Metadata samples:", len(meta))

# load expression
expr = pd.read_csv(EXPR, sep="\t", compression="gzip")
expr = expr.rename(columns={expr.columns[0]: "GeneID"})

print("Expression shape:", expr.shape)

# USE ALL SAMPLE COLUMNS DIRECTLY
sample_cols = expr.columns[1:]

expr = expr[["GeneID"] + list(sample_cols)]

# transpose
X = expr.set_index("GeneID").T
X.index.name = "sample_id"

print("X before filtering:", X.shape)

# keep only samples that exist in metadata
X = X.loc[X.index.isin(meta["sample_id"])]

print("X after matching metadata:", X.shape)

# log transform
X = np.log2(X + 1)

# remove low variance genes
variances = X.var(axis=0)
X = X.loc[:, variances > 0.01]

# select top genes
top_genes = X.var(axis=0).sort_values(ascending=False).head(500).index
X = X[top_genes]

# scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    index=X.index,
    columns=X.columns
)

# labels
y = meta.set_index("sample_id").loc[X_scaled.index, "label"]
y = y.map({"CTRL": 0, "AD": 1})

# save
X_scaled.to_csv(PROCESSED / "X_ad_ctrl.csv")
y.to_csv(PROCESSED / "y_ad_ctrl.csv", header=["label"])

print("\n✅ DATASET CREATED")
print("X shape:", X_scaled.shape)
print("Labels:")
print(y.value_counts())
