# ==========================================
# 🧬 Protein Allergenicity Predictor (Streamlit)
# – Uses FAIR’s ESM-2 (1280-dim) + XGBoost
# – Includes runtime hacks to avoid MKL/OpenMP & numpy._core issues
# ==========================================

# 1) ENVIRONMENT VARIABLE HACKS (must be first!)
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"

# 2) STANDARD IMPORTS
import streamlit as st
import torch
import joblib
import matplotlib.pyplot as plt

from esm import pretrained


# ==========================================
# === 4) MODEL & TOKENIZER LOADING
# ==========================================
@st.cache_resource
def load_esm2_model():
    model, alphabet = pretrained.esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


@st.cache_resource
def load_xgb_model(path):
    return joblib.load(path)


# ==========================================
# === 5) PREDICTION FUNCTION
# ==========================================
def predict(sequence, esm_model, batch_converter, xgb_model):
    seq = "".join(filter(str.isalpha, sequence.strip().upper()))
    valid = set("ACDEFGHIKLMNPQRSTVWY")

    if not seq:
        return "⚠️ Please enter a protein sequence.", None
    if any(aa not in valid for aa in seq):
        return "❌ Invalid sequence: use only 20 standard amino acids.", None
    if not (5 <= len(seq) <= 2000):
        return "⚠️ Sequence length must be 5–2000 AA.", None

    # Get [CLS] embedding
    batch = [("prot", seq)]
    _, _, tokens = batch_converter(batch)
    with torch.no_grad():
        out = esm_model(tokens, repr_layers=[33], return_contacts=False)
    emb = out["representations"][33][0, 0, :].cpu().numpy().reshape(1, -1)

    # Predict
    prob = xgb_model.predict_proba(emb)[0, 1]
    pred = xgb_model.predict(emb)[0]
    label = "🟢 Allergen" if pred else "🔴 Non-Allergen"
    result = f"**Label:** {label}  \n**P(Allergen):** `{prob:.4f}`"

    return result, emb


# ==========================================
# === 6) STREAMLIT APP
# ==========================================
def main():
    st.set_page_config(page_title="Protein Allergenicity Predictor", layout="centered")
    st.title("🧬 Protein Allergenicity Predictor")
    st.markdown("Paste a protein sequence (A,C,D,...,Y) and click **Predict**.")

    seq_in = st.text_area("Protein Sequence", height=150)

    # Lazy-load models
    esm_model, batch_converter = load_esm2_model()
    xgb_model = load_xgb_model("XGBoost_ESM-2_1280dim_algpred2_xgboost_model.pkl")

    if st.button("Predict Allergenicity"):
        result, emb = predict(seq_in, esm_model, batch_converter, xgb_model)
        st.markdown(result)


if __name__ == "__main__":
    main()
