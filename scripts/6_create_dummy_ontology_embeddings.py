import os
import json
import numpy as np

OUTPUT_DIR = "data/ontology"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBED_DIM = 32
np.random.seed(42)

concepts = [
    "RDP_Port_3389",
    "HTTP_Port_80",
    "HTTPS_Port_443",
    "SSH_Port_22",
    "DNS_Port_53",
    "FTP_Port_21",
    "SMTP_Port_25",
    "UNKNOWN_PORT"
]

embeddings = {}
for concept in concepts:
    embeddings[concept] = np.random.normal(0, 1, EMBED_DIM).tolist()

with open(os.path.join(OUTPUT_DIR, "ontology_embeddings.json"), "w") as f:
    json.dump(embeddings, f, indent=2)

print("Saved ontology embeddings to data/ontology/ontology_embeddings.json")
print("Concepts:", concepts)