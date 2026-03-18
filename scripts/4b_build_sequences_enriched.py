import os
import numpy as np

DATA_DIR = "data/enriched"
OUTPUT_DIR = "data/enriched_sequences"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 15

print("Loading enriched arrays...")
X_train = np.load(os.path.join(DATA_DIR, "X_train_enriched.npy"), mmap_mode="r")
X_test = np.load(os.path.join(DATA_DIR, "X_test_enriched.npy"), mmap_mode="r")
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"), mmap_mode="r")
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"), mmap_mode="r")

print("Original enriched shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

def build_sequences_memmap(X, y, seq_len, x_out_path, y_out_path):
    n_seq = len(X) - seq_len + 1
    feat_dim = X.shape[1]

    print(f"\nCreating memmap for {x_out_path}")
    print(f"Sequence count: {n_seq}, seq_len: {seq_len}, feat_dim: {feat_dim}")

    X_seq = np.lib.format.open_memmap(
        x_out_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_seq, seq_len, feat_dim)
    )

    y_seq = np.lib.format.open_memmap(
        y_out_path,
        mode="w+",
        dtype=np.int64,
        shape=(n_seq,)
    )

    for i in range(n_seq):
        X_seq[i] = X[i:i+seq_len]
        y_seq[i] = 1 if np.any(y[i:i+seq_len] == 1) else 0

        if i % 50000 == 0:
            print(f"Processed {i}/{n_seq}")

    X_seq.flush()
    y_seq.flush()

    return X_seq.shape, y_seq.shape

train_x_path = os.path.join(OUTPUT_DIR, "X_train_enriched_seq.npy")
train_y_path = os.path.join(OUTPUT_DIR, "y_train_enriched_seq.npy")
test_x_path = os.path.join(OUTPUT_DIR, "X_test_enriched_seq.npy")
test_y_path = os.path.join(OUTPUT_DIR, "y_test_enriched_seq.npy")

print("\nBuilding enriched train sequences...")
train_x_shape, train_y_shape = build_sequences_memmap(
    X_train, y_train, SEQ_LEN, train_x_path, train_y_path
)

print("\nBuilding enriched test sequences...")
test_x_shape, test_y_shape = build_sequences_memmap(
    X_test, y_test, SEQ_LEN, test_x_path, test_y_path
)

print("\nEnriched sequence shapes:")
print("X_train_seq:", train_x_shape)
print("y_train_seq:", train_y_shape)
print("X_test_seq :", test_x_shape)
print("y_test_seq :", test_y_shape)

print("\nSaved enriched sequence files:")
print(" - data/enriched_sequences/X_train_enriched_seq.npy")
print(" - data/enriched_sequences/y_train_enriched_seq.npy")
print(" - data/enriched_sequences/X_test_enriched_seq.npy")
print(" - data/enriched_sequences/y_test_enriched_seq.npy")