import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================
# CONFIG
# =========================
DATA_DIR = "data/enriched_real_sequences"
RESULTS_DIR = "results/cnn_lstm_real_enriched_sampled"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_BENIGN = 100000
MAX_TEST = 100000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("Using device:", DEVICE)

# =========================
# LOAD DATA
# =========================
X_train = np.load(
    os.path.join(DATA_DIR, "X_train_real_enriched_seq.npy"),
    mmap_mode="r"
)
y_train = np.load(
    os.path.join(DATA_DIR, "y_train_real_enriched_seq.npy"),
    mmap_mode="r"
)
X_test = np.load(
    os.path.join(DATA_DIR, "X_test_real_enriched_seq.npy"),
    mmap_mode="r"
)
y_test = np.load(
    os.path.join(DATA_DIR, "y_test_real_enriched_seq.npy"),
    mmap_mode="r"
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# =========================
# SAMPLING
# =========================
benign_idx = np.where(y_train == 0)[0]

if len(benign_idx) > MAX_BENIGN:
    benign_idx = np.random.choice(benign_idx, MAX_BENIGN, replace=False)

test_idx = np.arange(len(X_test))
if len(test_idx) > MAX_TEST:
    test_idx = np.random.choice(test_idx, MAX_TEST, replace=False)

X_train = np.array(X_train[benign_idx], dtype=np.float32)
X_test = np.array(X_test[test_idx], dtype=np.float32)
y_test = np.array(y_test[test_idx], dtype=np.int64)

print("Sampled train:", X_train.shape)
print("Sampled test :", X_test.shape)

# =========================
# DATALOADER
# =========================
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train)),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test)),
    batch_size=BATCH_SIZE,
    shuffle=False
)

seq_len = X_train.shape[1]
input_dim = X_train.shape[2]

# =========================
# MODEL
# =========================
class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)

        self.lstm_enc = nn.LSTM(128, 64, batch_first=True)
        self.lstm_dec = nn.LSTM(64, 128, batch_first=True)

        self.fc = nn.Linear(128, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)

        enc, _ = self.lstm_enc(x)
        dec, _ = self.lstm_dec(enc)

        return self.fc(dec)

model = Model(input_dim).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# =========================
# TRAIN
# =========================
print("\nTraining...")

for epoch in range(EPOCHS):
    total = 0.0
    model.train()

    for (x,) in train_loader:
        x = x.to(DEVICE)

        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, x)
        loss.backward()
        opt.step()

        total += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total/len(train_loader):.6f}")

# Save model
model_path = os.path.join(RESULTS_DIR, "cnn_lstm_real_enriched_sampled_model.pt")
torch.save(model.state_dict(), model_path)
print("\nSaved trained model to:", model_path)

# =========================
# THRESHOLD
# =========================
model.eval()
train_err = []

with torch.no_grad():
    for (x,) in train_loader:
        x = x.to(DEVICE)
        out = model(x)
        err = ((out - x) ** 2).mean(dim=(1, 2))
        train_err.extend(err.cpu().numpy())

train_err = np.array(train_err)
threshold = np.percentile(train_err, 99.5)
print("\nThreshold:", threshold)

# =========================
# TEST
# =========================
test_err = []

with torch.no_grad():
    for (x,) in test_loader:
        x = x.to(DEVICE)
        out = model(x)
        err = ((out - x) ** 2).mean(dim=(1, 2))
        test_err.extend(err.cpu().numpy())

test_err = np.array(test_err)
y_pred = (test_err > threshold).astype(int)

# =========================
# METRICS
# =========================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n=== REAL ONTOLOGY RESULTS ===")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
print("Confusion Matrix:\n", cm)

# =========================
# SAVE OUTPUTS
# =========================
np.save(os.path.join(RESULTS_DIR, "train_errors.npy"), train_err)
np.save(os.path.join(RESULTS_DIR, "test_errors.npy"), test_err)
np.save(os.path.join(RESULTS_DIR, "y_test.npy"), y_test)
np.save(os.path.join(RESULTS_DIR, "y_pred.npy"), y_pred)

with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump({
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "threshold": float(threshold),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seq_len": int(seq_len),
        "input_dim": int(input_dim),
        "max_benign": int(MAX_BENIGN),
        "max_test": int(MAX_TEST)
    }, f, indent=4)

print("\nSaved to:", RESULTS_DIR)