import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
DATA_PATH = "data/cicids2017_full.pkl"
EMBED_FILE = "data/ontology/ontology_node2vec_embeddings.json"
OUTPUT_DIR = "data/enriched_real"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42

print("Loading CICIDS2017 raw dataset...")
df = pd.read_pickle(DATA_PATH)

print("Original shape:", df.shape)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print("Shape after removing NaN/inf:", df.shape)

if "Destination Port" not in df.columns:
    raise ValueError("Column 'Destination Port' not found.")

if "Label" not in df.columns:
    raise ValueError("Column 'Label' not found.")

with open(EMBED_FILE, "r") as f:
    ontology_embeddings = json.load(f)

# =========================
# HELPERS
# =========================
def find_embedding_key(port_value, ontology_embeddings):
    """
    Try to map common ports to ontology node names.
    This function searches keys that contain the port number or common service term.
    """
    port = int(round(float(port_value)))

    candidates = []

    if port == 3389:
        candidates = ["3389", "RDP", "RemoteDesktop"]
    elif port == 22:
        candidates = ["22", "SSH"]
    elif port == 80:
        candidates = ["80", "HTTP"]
    elif port == 443:
        candidates = ["443", "HTTPS"]
    elif port == 53:
        candidates = ["53", "DNS"]
    elif port == 21:
        candidates = ["21", "FTP"]
    elif port == 25:
        candidates = ["25", "SMTP"]
    elif port == 3306:
        candidates = ["3306", "MYSQL", "MySQL"]
    elif port == 5432:
        candidates = ["5432", "POSTGRES", "Postgres", "PostgreSQL"]
    else:
        candidates = ["UNKNOWN", "PORT"]

    keys = list(ontology_embeddings.keys())

    for key in keys:
        key_upper = key.upper()
        for c in candidates:
            if c.upper() in key_upper:
                return key

    # fallback: just return first key
    return keys[0]

# =========================
# BUILD EMBEDDING MATRIX
# =========================
raw_ports = df["Destination Port"].values

# determine embedding dimension from first entry
first_key = next(iter(ontology_embeddings))
embed_dim = len(ontology_embeddings[first_key])

embedding_matrix = np.zeros((len(df), embed_dim), dtype=np.float32)

print("\nBuilding real ontology embedding matrix...")
for i, port in enumerate(raw_ports):
    key = find_embedding_key(port, ontology_embeddings)
    embedding_matrix[i] = np.array(ontology_embeddings[key], dtype=np.float32)

    if i % 500000 == 0:
        print(f"Processed {i}/{len(df)} rows")

print("Ontology embedding matrix shape:", embedding_matrix.shape)

# =========================
# SCALE FLOW FEATURES
# =========================
X_flow = df.drop(columns=["Label"]).copy()
y = df["Label"].values

scaler = MinMaxScaler()
X_flow_scaled = scaler.fit_transform(X_flow)

print("Scaled flow feature shape:", X_flow_scaled.shape)

# =========================
# CONCATENATE
# =========================
X_enriched = np.concatenate([X_flow_scaled, embedding_matrix], axis=1)

print("\nFinal real-enriched feature shape:", X_enriched.shape)

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_enriched,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

print("\nTrain/Test shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# =========================
# SAVE
# =========================
np.save(os.path.join(OUTPUT_DIR, "X_train_real_enriched.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "X_test_real_enriched.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print("\nSaved real-enriched arrays:")
print(" - data/enriched_real/X_train_real_enriched.npy")
print(" - data/enriched_real/X_test_real_enriched.npy")
print(" - data/enriched_real/y_train.npy")
print(" - data/enriched_real/y_test.npy")