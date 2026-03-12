import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

DATA_PATH = "data/cicids2017_full.pkl"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_pickle(DATA_PATH)

print("Original shape:", df.shape)

# Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum().sum())

# Replace inf with NaN, then drop NaN
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print("\nShape after removing NaN/inf:", df.shape)

# Split features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

print("\nFeature shape:", X.shape)
print("Label shape:", y.shape)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save full processed arrays
np.save(os.path.join(OUTPUT_DIR, "X_cicids.npy"), X_scaled)
np.save(os.path.join(OUTPUT_DIR, "y_cicids.npy"), y.values)

print("\nSaved full processed arrays:")
print(" - data/processed/X_cicids.npy")
print(" - data/processed/y_cicids.npy")

# Optional train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values,
    test_size=0.2,
    random_state=42,
    stratify=y.values
)

np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print("\nSaved train/test split:")
print(" - data/processed/X_train.npy")
print(" - data/processed/X_test.npy")
print(" - data/processed/y_train.npy")
print(" - data/processed/y_test.npy")

print("\nTrain/Test shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\nLabel distribution:")
unique, counts = np.unique(y.values, return_counts=True)
for label, count in zip(unique, counts):
    print(f"Label {label}: {count}")