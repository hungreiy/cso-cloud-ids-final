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
OUTPUT_DIR = "data/enriched"
ONTOLOGY_DIR = "data/ontology"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ONTOLOGY_DIR, exist_ok=True)

EMBED_DIM = 32
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# =========================
# DEFINE ONTOLOGY CONCEPTS
# =========================
CONCEPTS = [
    "RDP_Port_3389",
    "HTTP_Port_80",
    "HTTPS_Port_443",
    "SSH_Port_22",
    "DNS_Port_53",
    "FTP_Port_21",
    "SMTP_Port_25",
    "MYSQL_Port_3306",
    "POSTGRES_Port_5432",
    "UNKNOWN_PORT"
]

# =========================
# CREATE / LOAD ONTOLOGY EMBEDDINGS
# =========================
ONTOLOGY_FILE = os.path.join(ONTOLOGY_DIR, "ontology_embeddings.json")

if os.path.exists(ONTOLOGY_FILE):
    with open(ONTOLOGY_FILE, "r") as f:
        ontology_embeddings = json.load(f)
    print("Loaded existing ontology embeddings from:", ONTOLOGY_FILE)
else:
    ontology_embeddings = {}
    for concept in CONCEPTS:
        ontology_embeddings[concept] = np.random.normal(0, 1, EMBED_DIM).tolist()

    with open(ONTOLOGY_FILE, "w") as f:
        json.dump(ontology_embeddings, f, indent=2)

    print("Created new ontology embeddings at:", ONTOLOGY_FILE)

# =========================
# HELPER: MAP RAW PORT TO ONTOLOGY CONCEPT
# =========================
def map_port_to_concept(port_value):
    try:
        port = int(round(float(port_value)))
    except Exception:
        return "UNKNOWN_PORT"

    if port == 3389:
        return "RDP_Port_3389"
    elif port == 80:
        return "HTTP_Port_80"
    elif port == 443:
        return "HTTPS_Port_443"
    elif port == 22:
        return "SSH_Port_22"
    elif port == 53:
        return "DNS_Port_53"
    elif port == 21:
        return "FTP_Port_21"
    elif port == 25:
        return "SMTP_Port_25"
    elif port == 3306:
        return "MYSQL_Port_3306"
    elif port == 5432:
        return "POSTGRES_Port_5432"
    else:
        return "UNKNOWN_PORT"

# =========================
# LOAD RAW DATAFRAME
# =========================
print("\nLoading CICIDS2017 raw dataset...")
df = pd.read_pickle(DATA_PATH)

print("Original shape:", df.shape)

# =========================
# CLEAN DATA
# =========================
print("\nChecking missing values...")
print("Missing values before cleaning:", df.isnull().sum().sum())

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print("Shape after removing NaN/inf:", df.shape)

# =========================
# VERIFY REQUIRED COLUMNS
# =========================
if "Destination Port" not in df.columns:
    raise ValueError("Column 'Destination Port' not found in dataframe.")

if "Label" not in df.columns:
    raise ValueError("Column 'Label' not found in dataframe.")

# Save raw destination port before scaling
raw_ports = df["Destination Port"].copy()

# =========================
# BUILD ONTOLOGY EMBEDDING MATRIX FROM RAW PORTS
# =========================
print("\nBuilding ontology embedding matrix from raw Destination Port...")

port_concepts = raw_ports.apply(map_port_to_concept)

embedding_matrix = np.array(
    [ontology_embeddings[concept] for concept in port_concepts],
    dtype=np.float32
)

print("Ontology embedding matrix shape:", embedding_matrix.shape)

# =========================
# PREPARE FLOW FEATURES
# =========================
X_flow = df.drop(columns=["Label"]).copy()
y = df["Label"].values

print("Flow feature shape before scaling:", X_flow.shape)
print("Label shape:", y.shape)

# Scale only flow features
scaler = MinMaxScaler()
X_flow_scaled = scaler.fit_transform(X_flow)

print("Flow feature shape after scaling:", X_flow_scaled.shape)

# =========================
# CONCATENATE FLOW FEATURES + ONTOLOGY EMBEDDINGS
# =========================
X_enriched = np.concatenate([X_flow_scaled, embedding_matrix], axis=1)

print("\nFinal enriched feature shape:", X_enriched.shape)
print("Expected: 78 + 32 = 110 features")

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train_enriched, X_test_enriched, y_train, y_test = train_test_split(
    X_enriched,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

print("\nTrain/Test split:")
print("X_train_enriched:", X_train_enriched.shape)
print("X_test_enriched :", X_test_enriched.shape)
print("y_train         :", y_train.shape)
print("y_test          :", y_test.shape)

# =========================
# SAVE OUTPUTS
# =========================
np.save(os.path.join(OUTPUT_DIR, "X_train_enriched.npy"), X_train_enriched)
np.save(os.path.join(OUTPUT_DIR, "X_test_enriched.npy"), X_test_enriched)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print("\nSaved enriched arrays:")
print(" - data/enriched/X_train_enriched.npy")
print(" - data/enriched/X_test_enriched.npy")
print(" - data/enriched/y_train.npy")
print(" - data/enriched/y_test.npy")

# Optional: save concept distribution for inspection
concept_counts = port_concepts.value_counts().to_dict()
with open(os.path.join(OUTPUT_DIR, "ontology_concept_distribution.json"), "w") as f:
    json.dump(concept_counts, f, indent=2)

print("\nSaved ontology concept distribution:")
print(" - data/enriched/ontology_concept_distribution.json")

print("\nDone.")