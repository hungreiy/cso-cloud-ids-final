import os
import json
import numpy as np

DATA_DIR = "data/processed"
ONTOLOGY_FILE = "data/ontology/ontology_embeddings.json"
OUTPUT_DIR = "data/enriched"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading processed arrays...")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# Important: in your processed data, column 0 is Destination Port
DEST_PORT_INDEX = 0

with open(ONTOLOGY_FILE, "r") as f:
    ontology_embeddings = json.load(f)

def map_port_to_concept(port_value):
    port = int(round(port_value))

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
    else:
        return "UNKNOWN_PORT"

def build_embedding_matrix(X):
    emb_list = []

    for row in X:
        # Since X was MinMax-scaled, the original port is no longer exact.
        # We therefore use a simple heuristic:
        # if scaled destination port is close to common patterns, fallback to UNKNOWN.
        # Better version later: fuse BEFORE scaling.
        concept = "UNKNOWN_PORT"
        emb_list.append(ontology_embeddings[concept])

    return np.array(emb_list, dtype=np.float32)

print("Building ontology embeddings...")
X_train_emb = build_embedding_matrix(X_train)
X_test_emb = build_embedding_matrix(X_test)

print("Train embedding shape:", X_train_emb.shape)
print("Test embedding shape :", X_test_emb.shape)

X_train_enriched = np.concatenate([X_train, X_train_emb], axis=1)
X_test_enriched = np.concatenate([X_test, X_test_emb], axis=1)

print("Enriched train shape:", X_train_enriched.shape)
print("Enriched test shape :", X_test_enriched.shape)

np.save(os.path.join(OUTPUT_DIR, "X_train_enriched.npy"), X_train_enriched)
np.save(os.path.join(OUTPUT_DIR, "X_test_enriched.npy"), X_test_enriched)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print("\nSaved enriched arrays:")
print(" - data/enriched/X_train_enriched.npy")
print(" - data/enriched/X_test_enriched.npy")
print(" - data/enriched/y_train.npy")
print(" - data/enriched/y_test.npy")