# ==========================================
# 🧬 Protein Allergenicity Predictor (Streamlit)
# – Uses FAIR’s ESM-2 (1280-dim) + XGBoost + SHAP
# – Includes runtime hacks to avoid MKL/OpenMP & numpy._core issues
# ==========================================

# 1) ENVIRONMENT VARIABLE HACKS (must be first!)
import os, sys
# Allow duplicate OpenMP runtimes (Intel MKL vs PyTorch’s)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"]     = "1"

# 2) numpy._core ALIAS (before any joblib.load / pickle-based imports)
import numpy as _np
sys.modules['numpy._core'] = _np.core

# 3) STANDARD IMPORTS
import streamlit as st
import torch
import joblib
import shap
import matplotlib.pyplot as plt

from esm import pretrained

# ==========================================
# === 4) MODEL & TOKENIZER LOADING
# ==========================================
@st.cache_resource
def load_esm2_model():
    # Uses the 1280-dim ESM-2 model
    model, alphabet = pretrained.esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

@st.cache_resource
def load_xgb_model(path):
    return joblib.load(path)

@st.cache_resource
def load_shap_explainer(xgb_pipeline):
    # TreeExplainer under the hood for XGBoost
    return shap.Explainer(xgb_pipeline)

# ==========================================
# === 5) PREDICTION & SHAP FUNCTIONS
# ==========================================
def predict(sequence, esm_model, batch_converter, xgb_model, explainer):
    seq = ''.join(filter(str.isalpha, sequence.strip().upper()))
    valid = set("ACDEFGHIKLMNPQRSTVWY")

    if not seq:
        return "⚠️ Please enter a protein sequence.", None, None
    if any(aa not in valid for aa in seq):
        return "❌ Invalid sequence: use only 20 standard amino acids.", None, None
    if not (5 <= len(seq) <= 2000):
        return "⚠️ Sequence length must be 5–2000 AA.", None, None

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

    # SHAP
    shap_vals = explainer(emb)
    return result, shap_vals, emb

def plot_shap(shap_vals):
    plt.figure(figsize=(8,4))
    shap.plots.bar(shap_vals, max_display=10, show=False)
    st.pyplot(plt)

def plot_top_dims(shap_vals):
    vals = shap_vals.values[0]
    idx = _np.argsort(_np.abs(vals))[-10:][::-1]
    dims = [f"Dim {i}" for i in idx]
    cont = vals[idx]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(dims, cont)
    ax.invert_yaxis()
    ax.set_title("Top 10 SHAP-Contributing Dimensions")
    st.pyplot(fig)

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
    explainer  = load_shap_explainer(xgb_model)

    if st.button("Predict Allergenicity"):
        result, shap_vals, emb = predict(seq_in, esm_model, batch_converter, xgb_model, explainer)
        st.markdown(result)
        if shap_vals is not None:
            st.subheader("🧠 SHAP Explainability")
            plot_shap(shap_vals)
            plot_top_dims(shap_vals)

if __name__ == "__main__":
    main()
