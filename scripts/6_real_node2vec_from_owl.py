import os
import json
import random
from collections import defaultdict

import numpy as np
import networkx as nx
from rdflib import Graph
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# =========================
# CONFIG
# =========================
OWL_PATH = "data/ontology/CloudSecurityOntology.owl"
OUTPUT_DIR = "data/ontology"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBED_DIM = 32
WALK_LENGTH = 10
NUM_WALKS = 20
WINDOW_SIZE = 5
P = 0.5   # return parameter
Q = 0.5   # in-out parameter
SEED = 42

EMB_JSON = os.path.join(OUTPUT_DIR, "ontology_node2vec_embeddings.json")
NODES_JSON = os.path.join(OUTPUT_DIR, "ontology_nodes.json")

random.seed(SEED)
np.random.seed(SEED)

print("Loading OWL ontology from:", OWL_PATH)

if not os.path.exists(OWL_PATH):
    raise FileNotFoundError(f"Ontology file not found: {OWL_PATH}")

# =========================
# LOAD RDF / OWL
# =========================
rdf_graph = Graph()
rdf_graph.parse(OWL_PATH)

print("Total RDF triples loaded:", len(rdf_graph))

# =========================
# BUILD NETWORKX GRAPH
# =========================
G = nx.Graph()

for subj, pred, obj in rdf_graph:
    s = str(subj)
    o = str(obj)

    # Skip pure literals that explode graph noise
    if not (
        o.startswith("http://")
        or o.startswith("https://")
        or o.startswith("file:")
        or o.startswith("urn:")
    ):
        continue

    G.add_node(s)
    G.add_node(o)
    G.add_edge(s, o)

print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())

if G.number_of_nodes() == 0:
    raise ValueError("No usable graph nodes were created from the ontology.")

# =========================
# NODE2VEC-STYLE BIASED RANDOM WALK
# =========================
def biased_next_step(graph, prev_node, curr_node, p=1.0, q=1.0):
    neighbors = list(graph.neighbors(curr_node))
    if not neighbors:
        return None

    weights = []
    for dst in neighbors:
        if prev_node is None:
            weight = 1.0
        elif dst == prev_node:
            weight = 1.0 / p
        elif graph.has_edge(dst, prev_node):
            weight = 1.0
        else:
            weight = 1.0 / q
        weights.append(weight)

    weights = np.array(weights, dtype=np.float64)
    probs = weights / weights.sum()
    return np.random.choice(neighbors, p=probs)

def generate_walk(graph, start_node, walk_length, p=1.0, q=1.0):
    walk = [start_node]

    while len(walk) < walk_length:
        curr = walk[-1]
        prev = walk[-2] if len(walk) > 1 else None

        nxt = biased_next_step(graph, prev, curr, p=p, q=q)
        if nxt is None:
            break

        walk.append(nxt)

    return walk

print("\nGenerating random walks...")
nodes = list(G.nodes())
walks = []

for n_round in range(NUM_WALKS):
    random.shuffle(nodes)
    for node in nodes:
        walk = generate_walk(G, node, WALK_LENGTH, p=P, q=Q)
        if len(walk) >= 2:
            walks.append(walk)

    print(f"Completed walk round {n_round + 1}/{NUM_WALKS}")

print("Total walks generated:", len(walks))

# =========================
# BUILD NODE INDEX
# =========================
all_nodes = list(G.nodes())
node_to_idx = {node: i for i, node in enumerate(all_nodes)}
num_nodes = len(all_nodes)

# =========================
# BUILD CO-OCCURRENCE MATRIX FROM WALKS
# =========================
print("\nBuilding co-occurrence matrix...")
pair_counts = defaultdict(float)

for walk in walks:
    indexed_walk = [node_to_idx[w] for w in walk if w in node_to_idx]

    for i, center in enumerate(indexed_walk):
        left = max(0, i - WINDOW_SIZE)
        right = min(len(indexed_walk), i + WINDOW_SIZE + 1)

        for j in range(left, right):
            if i == j:
                continue
            context = indexed_walk[j]
            pair_counts[(center, context)] += 1.0

rows = []
cols = []
vals = []

for (r, c), v in pair_counts.items():
    rows.append(r)
    cols.append(c)
    vals.append(v)

cooc = coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)

print("Co-occurrence matrix shape:", cooc.shape)
print("Non-zero entries:", cooc.nnz)

# =========================
# TRUNCATED SVD EMBEDDING
# =========================
print("\nRunning TruncatedSVD for embeddings...")

n_components = min(EMBED_DIM, max(2, num_nodes - 1))

svd = TruncatedSVD(n_components=n_components, random_state=SEED)
emb = svd.fit_transform(cooc)

# If graph is too small, pad dimensions
if emb.shape[1] < EMBED_DIM:
    pad_width = EMBED_DIM - emb.shape[1]
    emb = np.pad(emb, ((0, 0), (0, pad_width)), mode="constant")

# Normalize embeddings
emb = normalize(emb)

# =========================
# SAVE OUTPUTS
# =========================
embeddings = {}
for i, node in enumerate(all_nodes):
    embeddings[node] = emb[i].tolist()

with open(EMB_JSON, "w") as f:
    json.dump(embeddings, f, indent=2)

with open(NODES_JSON, "w") as f:
    json.dump(all_nodes, f, indent=2)

print("\nSaved ontology embeddings:")
print(" -", EMB_JSON)
print(" -", NODES_JSON)

print("\nSample nodes:")
for node in all_nodes[:10]:
    print(node)