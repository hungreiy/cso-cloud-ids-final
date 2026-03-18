import os
import numpy as np

DATA_DIR = "data/enriched_real"
OUTPUT_DIR = "data/enriched_real_sequences"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 15

X_train = np.load(os.path.join(DATA_DIR, "X_train_real_enriched.npy"), mmap_mode="r")
X_test = np.load(os.path.join(DATA_DIR, "X_test_real_enriched.npy"), mmap_mode="r")
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"), mmap_mode="r")
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"), mmap_mode="r")

print("Original shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

def build_sequences_memmap(X, y, seq_len, x_out_path, y_out_path):
    n_seq = len(X) - seq_len + 1
    feat_dim = X.shape[1]

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

train_x = os.path.join(OUTPUT_DIR, "X_train_real_enriched_seq.npy")
train_y = os.path.join(OUTPUT_DIR, "y_train_real_enriched_seq.npy")
test_x = os.path.join(OUTPUT_DIR, "X_test_real_enriched_seq.npy")
test_y = os.path.join(OUTPUT_DIR, "y_test_real_enriched_seq.npy")

print("\nBuilding train sequences...")
build_sequences_memmap(X_train, y_train, SEQ_LEN, train_x, train_y)

print("\nBuilding test sequences...")
build_sequences_memmap(X_test, y_test, SEQ_LEN, test_x, test_y)

print("\nSaved real enriched sequence files.")