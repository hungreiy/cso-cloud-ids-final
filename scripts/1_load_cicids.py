import pandas as pd

DATA_PATH = "data/cicids2017_full.pkl"

print("Loading CICIDS2017 dataset...")

df = pd.read_pickle(DATA_PATH)

print("\nDataset loaded successfully")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst rows:")
print(df.head())

print("\nLabel distribution:")
print(df['Label'].value_counts())