from pathlib import Path
import urllib.request
from urllib.parse import urlencode

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

def geo_download_url(acc: str, filename: str) -> str:
    params = urlencode({
        "acc": acc,
        "format": "file",
        "file": filename
    })
    return f"https://www.ncbi.nlm.nih.gov/geo/download/?{params}"

files = {
    "GSE161199_series_matrix.txt.gz":
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161199/matrix/GSE161199_series_matrix.txt.gz",

    "GSE161199_RAW.tar":
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161199/suppl/GSE161199_RAW.tar",

    "GSE161199_raw_counts_GRCh38.p13_NCBI.tsv.gz":
        geo_download_url("GSE161199", "GSE161199_raw_counts_GRCh38.p13_NCBI.tsv.gz"),

    "GSE161199_norm_counts_FPKM_GRCh38.p13_NCBI.tsv.gz":
        geo_download_url("GSE161199", "GSE161199_norm_counts_FPKM_GRCh38.p13_NCBI.tsv.gz"),

    "Human.GRCh38.p13.annot.tsv.gz":
        geo_download_url("GSE161199", "Human.GRCh38.p13.annot.tsv.gz"),
}

for filename, url in files.items():
    out = RAW / filename

    if out.exists() and out.stat().st_size > 0:
        print(f"Already exists: {out}")
        continue

    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, out)
    print(f"Saved to {out}")

print("Done.")
