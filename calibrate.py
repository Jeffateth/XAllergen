# calibrate.py

import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.isotonic import IsotonicRegression
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------
# Configuration
# -----------------------
MODEL_PATH = "allergenicity-webapp-streamlit/model"
CALIBRATOR_OUT = "allergenicity-webapp-streamlit/model/calibrator.joblib"
CALIB_PROBS_OUT = "allergenicity-webapp-streamlit/model/val_calibrated_probs.npy"
BATCH_SIZE = 16
MAX_LEN = 1024

# GitHub data source
CSV_URL = "https://raw.githubusercontent.com/Jeffateth/AllergenPredict/main/algpred2_test.csv"

# -----------------------
# Load model & tokenizer
# -----------------------
print("🔄 Loading model from", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

# -----------------------
# Load validation data from GitHub
# -----------------------
print(f"🌐 Downloading validation data from GitHub...")
df = pd.read_csv(CSV_URL)
sequences = df["sequence"].astype(str).tolist()
labels = df["label"].astype(int).values

# -----------------------
# Run inference on validation set
# -----------------------
print(f"📊 Running inference on {len(sequences)} sequences...")
raw_probs = []

for i in range(0, len(sequences), BATCH_SIZE):
    batch = sequences[i:i + BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[:, 1]  # allergen class
        raw_probs.extend(probs.numpy().tolist())

raw_probs = np.array(raw_probs)

# -----------------------
# Fit calibrator
# -----------------------
print("⚖️ Fitting IsotonicRegression calibrator...")
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(raw_probs, labels)

# -----------------------
# Save calibrator & calibrated values
# -----------------------
joblib.dump(calibrator, CALIBRATOR_OUT)
calibrated_probs = calibrator.transform(raw_probs)
np.save(CALIB_PROBS_OUT, calibrated_probs)

print(f"\n✅ Saved calibrator → {CALIBRATOR_OUT}")
print(f"✅ Saved calibrated probabilities → {CALIB_PROBS_OUT}")
