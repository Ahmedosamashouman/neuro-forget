from pathlib import Path
import gzip
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/interim")
OUT.mkdir(parents=True, exist_ok=True)

SERIES = RAW / "GSE161199_series_matrix.txt.gz"

sample_ids = []
titles = []

with gzip.open(SERIES, "rt", errors="ignore") as f:
    for line in f:
        if line.startswith("!Sample_geo_accession"):
            sample_ids = line.strip().split("\t")[1:]
        elif line.startswith("!Sample_title"):
            titles = line.strip().split("\t")[1:]

# clean values
sample_ids = [x.replace('"', '').strip() for x in sample_ids]
titles = [x.replace('"', '').strip() for x in titles]

def get_label(title):
    t = title.lower()
    if "ad" in t:
        return "AD"
    elif "control" in t or "ctrl" in t:
        return "CTRL"
    elif "pd" in t:
        return "PD"
    elif "als" in t:
        return "ALS"
    else:
        return "UNKNOWN"

meta = pd.DataFrame({
    "sample_id": sample_ids,
    "title": titles
})

meta["label"] = meta["title"].apply(get_label)

meta.to_csv(OUT / "metadata.csv", index=False)

print("Metadata created!")
print(meta.head())
print("\nLabel counts:")
print(meta["label"].value_counts())
