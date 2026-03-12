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
DATA_DIR = "data/processed"
RESULTS_DIR = "results/baseline"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# =========================
# LOAD DATA
# =========================
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# Train autoencoder only on benign samples
X_train_benign = X_train[y_train == 0]
print("Benign training samples:", X_train_benign.shape)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_benign, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=BATCH_SIZE, shuffle=False)

input_dim = X_train.shape[1]

# =========================
# MODEL
# =========================
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

model = Autoencoder(input_dim).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================
# TRAIN
# =========================
print("\nTraining baseline autoencoder...")
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch in train_loader:
        x = batch[0].to(DEVICE)

        optimizer.zero_grad()
        recon = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")

# =========================
# THRESHOLD FROM TRAIN BENIGN ERRORS
# =========================
model.eval()
with torch.no_grad():
    train_recon = model(X_train_tensor.to(DEVICE)).cpu().numpy()

train_errors = np.mean((X_train_benign - train_recon) ** 2, axis=1)
threshold = np.percentile(train_errors, 99.5)

print("\nThreshold:", threshold)

# =========================
# EVALUATE
# =========================
all_test_recon = []
with torch.no_grad():
    for batch in test_loader:
        x = batch[0].to(DEVICE)
        recon = model(x).cpu().numpy()
        all_test_recon.append(recon)

X_test_recon = np.vstack(all_test_recon)
test_errors = np.mean((X_test - X_test_recon) ** 2, axis=1)

y_pred = (test_errors > threshold).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n=== BASELINE RESULTS ===")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
print("Confusion Matrix:\n", cm)

# =========================
# SAVE RESULTS
# =========================
results = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "threshold": float(threshold),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE
}

with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=4)

np.save(os.path.join(RESULTS_DIR, "test_errors.npy"), test_errors)
np.save(os.path.join(RESULTS_DIR, "y_pred.npy"), y_pred)
np.save(os.path.join(RESULTS_DIR, "y_test.npy"), y_test)

print("\nSaved results to:", RESULTS_DIR)