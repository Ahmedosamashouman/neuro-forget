from pathlib import Path
import random
import itertools
import pandas as pd

OUT = Path("data/fasta_demo")
OUT.mkdir(parents=True, exist_ok=True)

FASTA_PATH = OUT / "alzheimers_simulated.fasta"
META_PATH = OUT / "fasta_metadata.csv"
X_PATH = OUT / "X_kmers.csv"
Y_PATH = OUT / "y_kmers.csv"

random.seed(42)

NUM_SAMPLES = 200
SEQ_LENGTH = 150
K = 3

AD_MARKERS = ["TCCG", "CTCA"]

def random_dna(n):
    return "".join(random.choice("ACGT") for _ in range(n))

def inject_marker(seq, marker):
    pos = random.randint(0, len(seq) - len(marker))
    return seq[:pos] + marker + seq[pos + len(marker):]

records = []
meta = []

for i in range(NUM_SAMPLES):
    patient_id = f"Patient_{i:04d}"
    label = 1 if i < NUM_SAMPLES // 2 else 0

    seq = random_dna(SEQ_LENGTH)

    if label == 1:
        for marker in AD_MARKERS:
            if random.random() < 0.85:
                seq = inject_marker(seq, marker)

    records.append((patient_id, seq))
    meta.append({"patient_id": patient_id, "label": label})

with open(FASTA_PATH, "w") as f:
    for patient_id, seq in records:
        f.write(f">{patient_id}\n{seq}\n")

kmers = ["".join(p) for p in itertools.product("ACGT", repeat=K)]

def count_kmers(seq):
    counts = {k: 0 for k in kmers}
    for i in range(len(seq) - K + 1):
        kmer = seq[i:i+K]
        counts[kmer] += 1
    return counts

X_rows = []
for patient_id, seq in records:
    row = count_kmers(seq)
    row["patient_id"] = patient_id
    X_rows.append(row)

X = pd.DataFrame(X_rows).set_index("patient_id")
y = pd.DataFrame(meta).set_index("patient_id")

X.to_csv(X_PATH)
y.to_csv(Y_PATH)
pd.DataFrame(meta).to_csv(META_PATH, index=False)

print("FASTA demo generated")
print("FASTA:", FASTA_PATH)
print("X_kmers:", X.shape)
print("Labels:")
print(y["label"].value_counts())
