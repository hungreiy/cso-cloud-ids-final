import os
import numpy as np

DATA_DIR = "data/processed"
OUTPUT_DIR = "data/sequences"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 15

print("Loading processed arrays...")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("Original shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

def build_sequences(X, y, seq_len):
    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_len + 1):
        window_x = X[i:i+seq_len]
        window_y = y[i:i+seq_len]

        # label window as attack if any element in window is attack
        label = 1 if np.any(window_y == 1) else 0

        X_seq.append(window_x)
        y_seq.append(label)

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)

print("\nBuilding train sequences...")
X_train_seq, y_train_seq = build_sequences(X_train, y_train, SEQ_LEN)

print("Building test sequences...")
X_test_seq, y_test_seq = build_sequences(X_test, y_test, SEQ_LEN)

print("\nSequence shapes:")
print("X_train_seq:", X_train_seq.shape)
print("y_train_seq:", y_train_seq.shape)
print("X_test_seq :", X_test_seq.shape)
print("y_test_seq :", y_test_seq.shape)

np.save(os.path.join(OUTPUT_DIR, "X_train_seq.npy"), X_train_seq)
np.save(os.path.join(OUTPUT_DIR, "y_train_seq.npy"), y_train_seq)
np.save(os.path.join(OUTPUT_DIR, "X_test_seq.npy"), X_test_seq)
np.save(os.path.join(OUTPUT_DIR, "y_test_seq.npy"), y_test_seq)

print("\nSaved sequence files:")
print(" - data/sequences/X_train_seq.npy")
print(" - data/sequences/y_train_seq.npy")
print(" - data/sequences/X_test_seq.npy")
print(" - data/sequences/y_test_seq.npy")