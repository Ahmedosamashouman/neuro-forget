from pathlib import Path
import gzip
import pandas as pd

RAW = Path("data/gse63060/raw")
OUT = Path("data/gse63060/interim")
OUT.mkdir(parents=True, exist_ok=True)

FILE = RAW / "GSE63060_series_matrix.txt.gz"

sample_ids = []
titles = []

with gzip.open(FILE, "rt", errors="ignore") as f:
    for line in f:
        if line.startswith("!Sample_geo_accession"):
            sample_ids = line.strip().split("\t")[1:]
        elif line.startswith("!Sample_characteristics_ch1"):
            titles.append(line.strip().split("\t")[1:])

# flatten
titles = list(zip(*titles))
titles = [" ".join(t).lower() for t in titles]

def get_label(text):
    if "alzheimer" in text or "ad" in text:
        return "AD"
    elif "control" in text:
        return "CTRL"
    else:
        return "OTHER"

meta = pd.DataFrame({
    "sample_id": sample_ids,
    "text": titles
})

meta["label"] = meta["text"].apply(get_label)

meta = meta[meta["label"].isin(["AD", "CTRL"])]

meta.to_csv(OUT / "metadata.csv", index=False)

print("Metadata created")
print(meta["label"].value_counts())
