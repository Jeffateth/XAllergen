#!/usr/bin/env python3
"""
preprocess_allergen_data.py

Correctly preprocess allergen data for CNN model without data leakage:
- Loads training and test data separately
- Maintains original train/test split (no reshuffling)
- Applies StandardScaler fitted only on training data
- One-hot encodes sequences with consistent dimensions
- Saves preprocessed arrays to .npy files
"""
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# === Paths ===
DATA_DIR = "/Users/jianzhouyao/AllergenPredict/data"
OUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("[Start] Loading data files...")

# === 1) Load train and test CSVs separately ===
# Structure data
structure_train = pd.read_csv(os.path.join(DATA_DIR, "global_3d_summary_train.csv"))
structure_test = pd.read_csv(os.path.join(DATA_DIR, "global_3d_summary_test.csv"))

# Extract ID from PDB_File column
structure_train["id"] = structure_train["PDB_File"].str.replace(".pdb", "", regex=False)
structure_test["id"] = structure_test["PDB_File"].str.replace(".pdb", "", regex=False)

# Sequence data
seq_train = pd.read_csv(os.path.join(DATA_DIR, "algpred2_train_seq.csv"))
seq_test = pd.read_csv(os.path.join(DATA_DIR, "algpred2_test_seq.csv"))

print(f"[Data] Train sequences: {len(seq_train)}, Test sequences: {len(seq_test)}")

# === 2) Merge structure and sequence for train and test separately ===
train_df = pd.merge(seq_train, structure_train, on="id", how="inner")
test_df = pd.merge(seq_test, structure_test, on="id", how="inner")

# Drop PDB_File column
train_df = train_df.drop(columns=["PDB_File"])
test_df = test_df.drop(columns=["PDB_File"])

print(f"[Data] After merging - Train: {len(train_df)}, Test: {len(test_df)}")

# Check for any potential issues
train_ids = set(train_df["id"])
test_ids = set(test_df["id"])
overlap = train_ids.intersection(test_ids)
if overlap:
    print(f"[Warning] Found {len(overlap)} overlapping IDs between train and test!")
else:
    print("[Data] No overlapping IDs between train and test sets - good!")

# === 3) Normalize structure features (fit on train only) ===
struct_cols = [
    "Total_SASA",
    "Radius_of_Gyration",
    "Compactness",
    "Contact_Order",
    "SS_Helix",
    "SS_Strand",
    "SS_Coil",
]

print("[Process] Normalizing structural features...")
scaler = StandardScaler()
X_struct_train = scaler.fit_transform(train_df[struct_cols])
X_struct_test = scaler.transform(test_df[struct_cols])

# Save the scaler for future use
with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# === 4) One-hot encode sequences with consistent dimensions ===
print("[Process] One-hot encoding sequences...")
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

# Find the maximum sequence length across both datasets
max_len = max(train_df["sequence"].str.len().max(), test_df["sequence"].str.len().max())
print(f"[Process] Maximum sequence length: {max_len}")


def encode_seqs(seq_series):
    """One-hot encode amino acid sequences to a 3D tensor"""
    n = len(seq_series)
    X = np.zeros((n, len(AA_ALPHABET), max_len), dtype=np.float32)
    for i, seq in enumerate(seq_series):
        for pos, aa in enumerate(seq[:max_len]):
            idx = AA_TO_IDX.get(aa)
            if idx is not None:
                X[i, idx, pos] = 1.0
    return X


X_seq_train = encode_seqs(train_df["sequence"])
X_seq_test = encode_seqs(test_df["sequence"])

# === 5) Extract labels & ids ===
y_train = train_df["label"].values.astype(np.int64)
y_test = test_df["label"].values.astype(np.int64)
ids_train = train_df["id"].values
ids_test = test_df["id"].values

# Print class distribution
train_pos = sum(y_train)
test_pos = sum(y_test)
print(
    f"[Data] Train class distribution: {train_pos} allergens, {len(y_train)-train_pos} non-allergens"
)
print(
    f"[Data] Test class distribution: {test_pos} allergens, {len(y_test)-test_pos} non-allergens"
)

# === 6) Save preprocessed arrays to .npy files ===
print("[Save] Writing preprocessed arrays to disk...")
np.save(os.path.join(OUT_DIR, "X_seq_train.npy"), X_seq_train)
np.save(os.path.join(OUT_DIR, "X_seq_test.npy"), X_seq_test)
np.save(os.path.join(OUT_DIR, "X_struct_train.npy"), X_struct_train)
np.save(os.path.join(OUT_DIR, "X_struct_test.npy"), X_struct_test)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)
np.save(os.path.join(OUT_DIR, "ids_train.npy"), ids_train)
np.save(os.path.join(OUT_DIR, "ids_test.npy"), ids_test)

# Additional data verification
print("\n=== Data verification ===")
print(f"X_seq_train shape: {X_seq_train.shape}")
print(f"X_seq_test shape: {X_seq_test.shape}")
print(f"X_struct_train shape: {X_struct_train.shape}")
print(f"X_struct_test shape: {X_struct_test.shape}")

# Basic model input verification
assert X_seq_train.shape[0] == X_struct_train.shape[0] == y_train.shape[0]
assert X_seq_test.shape[0] == X_struct_test.shape[0] == y_test.shape[0]
print("✓ All array dimensions are consistent")

print("\n✓ Preprocessing complete; files written to", OUT_DIR)
