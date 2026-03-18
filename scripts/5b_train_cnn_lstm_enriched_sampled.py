import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATA_DIR = "data/enriched_sequences"
RESULTS_DIR = "results/cnn_lstm_enriched_sampled"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# sampling sizes
MAX_BENIGN_TRAIN = 100000
MAX_TEST = 100000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

print("Using device:", DEVICE)

X_train_seq = np.load(os.path.join(DATA_DIR, "X_train_enriched_seq.npy"), mmap_mode="r")
y_train_seq = np.load(os.path.join(DATA_DIR, "y_train_enriched_seq.npy"), mmap_mode="r")
X_test_seq = np.load(os.path.join(DATA_DIR, "X_test_enriched_seq.npy"), mmap_mode="r")
y_test_seq = np.load(os.path.join(DATA_DIR, "y_test_enriched_seq.npy"), mmap_mode="r")

print("Full enriched train shape:", X_train_seq.shape)
print("Full enriched test shape :", X_test_seq.shape)

# benign-only training
benign_idx = np.where(y_train_seq == 0)[0]
print("Total benign train sequences:", len(benign_idx))

if len(benign_idx) > MAX_BENIGN_TRAIN:
    benign_idx = np.random.choice(benign_idx, MAX_BENIGN_TRAIN, replace=False)

test_idx = np.arange(len(X_test_seq))
if len(test_idx) > MAX_TEST:
    test_idx = np.random.choice(test_idx, MAX_TEST, replace=False)

X_train_benign = np.array(X_train_seq[benign_idx], dtype=np.float32)
X_test_sample = np.array(X_test_seq[test_idx], dtype=np.float32)
y_test_sample = np.array(y_test_seq[test_idx], dtype=np.int64)

print("Sampled benign train shape:", X_train_benign.shape)
print("Sampled test shape        :", X_test_sample.shape)

X_train_tensor = torch.tensor(X_train_benign, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_sample, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=BATCH_SIZE, shuffle=False)

seq_len = X_train_benign.shape[1]
input_dim = X_train_benign.shape[2]

class CNNLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.lstm_encoder = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.lstm_decoder = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(128, input_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        encoded, _ = self.lstm_encoder(x)
        decoded, _ = self.lstm_decoder(encoded)
        out = self.output_layer(decoded)
        return out

model = CNNLSTMAutoencoder(input_dim=input_dim).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nTraining sampled ontology-enriched CNN-LSTM...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        x = batch[0].to(DEVICE)

        optimizer.zero_grad()
        recon = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "cnn_lstm_enriched_sampled_model.pt"))

# threshold
model.eval()
train_errors = []

with torch.no_grad():
    for batch in train_loader:
        x = batch[0].to(DEVICE)
        recon = model(x)
        err = torch.mean((recon - x) ** 2, dim=(1, 2))
        train_errors.extend(err.cpu().numpy())

train_errors = np.array(train_errors)
threshold = np.percentile(train_errors, 99.5)

print("\nThreshold:", threshold)

# test
test_errors = []

with torch.no_grad():
    for batch in test_loader:
        x = batch[0].to(DEVICE)
        recon = model(x)
        err = torch.mean((recon - x) ** 2, dim=(1, 2))
        test_errors.extend(err.cpu().numpy())

test_errors = np.array(test_errors)
y_pred = (test_errors > threshold).astype(int)

acc = accuracy_score(y_test_sample, y_pred)
prec = precision_score(y_test_sample, y_pred, zero_division=0)
rec = recall_score(y_test_sample, y_pred, zero_division=0)
f1 = f1_score(y_test_sample, y_pred, zero_division=0)
cm = confusion_matrix(y_test_sample, y_pred)

print("\n=== ENRICHED SAMPLED CNN-LSTM RESULTS ===")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
print("Confusion Matrix:\n", cm)

results = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "threshold": float(threshold),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "seq_len": int(seq_len),
    "input_dim": int(input_dim),
    "max_benign_train": int(MAX_BENIGN_TRAIN),
    "max_test": int(MAX_TEST)
}

with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=4)

np.save(os.path.join(RESULTS_DIR, "test_errors.npy"), test_errors)
np.save(os.path.join(RESULTS_DIR, "y_pred.npy"), y_pred)
np.save(os.path.join(RESULTS_DIR, "y_test.npy"), y_test_sample)

print("\nSaved results to:", RESULTS_DIR)