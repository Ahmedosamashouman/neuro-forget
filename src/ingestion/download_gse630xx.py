from pathlib import Path
import urllib.request

files = {
    "data/gse63060/raw/GSE63060_series_matrix.txt.gz":
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63060/matrix/GSE63060_series_matrix.txt.gz",

    "data/gse63061/raw/GSE63061_series_matrix.txt.gz":
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63061/matrix/GSE63061_series_matrix.txt.gz",
}

for out, url in files.items():
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print("Exists:", out)
        continue
    print("Downloading:", out)
    urllib.request.urlretrieve(url, out)

print("Done.")
