import re
import torch
import shap
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

from sklearn.calibration import calibration_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Constants
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 2000
AMINO_REGEX = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

CALIBRATOR_PATH = "model/calibrator.joblib"
VAL_CALIB_PROBS_PATH = "model/val_calibrated_probs.npy"
VALIDATION_LABELS_URL = "https://raw.githubusercontent.com/Jeffateth/AllergenPredict/main/algpred2_test.csv"

# Load model/tokenizer/explainer once
@st.cache_resource
def load_model():
    model_dir = "model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    explainer = shap.Explainer(model, tokenizer)
    return tokenizer, model, explainer

tokenizer, model, explainer = load_model()

# Load calibrator + validation probs
@st.cache_resource
def load_calibrator():
    try:
        calibrator = joblib.load(CALIBRATOR_PATH)
        val_probs = np.load(VAL_CALIB_PROBS_PATH)
        return calibrator, val_probs
    except Exception as e:
        st.warning(f"Calibration file not loaded: {e}")
        return None, None

calibrator, val_probs = load_calibrator()

# Title
st.title("🧪 Allergenicity Predictor (ESM-2)")

# Threshold slider
threshold = st.sidebar.slider(
    "Decision Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01
)

# Show calibration confidence distribution
if val_probs is not None:
    st.sidebar.markdown("**Confidence Histogram (Validation Set)**")
    st.sidebar.bar_chart(np.histogram(val_probs, bins=20)[0])

# Input form
seq_input = st.text_area(
    "Enter protein sequence (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y):",
    height=150
).strip().upper().replace(" ", "")

# Validation
if not seq_input:
    st.warning("Please enter a protein sequence.")
elif len(seq_input) < MIN_SEQ_LEN:
    st.error(f"Sequence too short (min {MIN_SEQ_LEN} AA).")
elif len(seq_input) > MAX_SEQ_LEN:
    st.error(f"Sequence too long (max {MAX_SEQ_LEN} AA).")
elif not AMINO_REGEX.match(seq_input):
    st.error("Invalid characters detected. Use only 20 standard amino acids.")
else:
    # Predict
    inputs = tokenizer(seq_input, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        raw_prob = float(probs[1])
        calibrated_prob = float(calibrator.transform([raw_prob])[0]) if calibrator else raw_prob

    label = "🟢 Allergen" if calibrated_prob > threshold else "🔴 Non-allergen"

    # Show result
    st.subheader("Prediction Result")
    st.write(f"**Label:** {label}")
    st.write(f"**Probability (Allergen):** {calibrated_prob:.4f}")
    st.caption(f"Raw model output: {raw_prob:.4f} → Calibrated: {calibrated_prob:.4f}")

    # SHAP Explanation
    st.subheader("SHAP Explanation")
    try:
        shap_result = explainer([seq_input])
        sv = shap_result.values
        if sv.ndim == 2:
            shap_vals = sv[:, 1]
        else:
            shap_vals = sv
        st.bar_chart(shap_vals)
    except Exception as e:
        st.error(f"SHAP explanation unavailable: {e}")

    # Calibration curve
    st.subheader("📈 Calibration Curve")
    try:
        y_true = pd.read_csv(VALIDATION_LABELS_URL)["label"].values
        y_pred = val_probs
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)

        fig, ax = plt.subplots()
        ax.plot(prob_pred, prob_true, marker='o', label="Calibrated")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("True frequency")
        ax.set_title("Calibration Curve (Validation Set)")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display calibration curve: {e}")
