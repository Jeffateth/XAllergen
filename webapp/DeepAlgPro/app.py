import streamlit as st
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

with st.expander("ℹ️ What this app does", expanded=False):
    st.markdown(
        """
    **DeepAlgPro** is a deep learning model trained to predict whether a protein sequence is allergenic.

    This web app allows you to:
    - Paste a protein sequence (1-letter code).
    - Get a prediction (allergenic or not).
    - Visualize how the model interprets the sequence using a self-attention heatmap.
    - See which residues the model considers most important.

    The model is based on a CNN + Multi-Head Self-Attention architecture and is pre-trained on a curated allergen dataset.
    """
    )


# Import from predict_with_attention instead of predict
from predict_with_attention import load_model, predict_single


# Load model (only once)
@st.cache_resource
def get_model():
    return load_model("model.pt")


model, device = get_model()

# Streamlit UI
st.title("🔬 DeepAlgPro: Allergenicity Prediction with Attention")
st.markdown(
    """
Paste your protein sequence below (1-letter code, max 1000 characters).  
We'll predict if it's allergenic and visualize the model's attention.
"""
)

# Input box
sequence = st.text_area("Enter Protein Sequence:", height=200)

if st.button("Predict"):
    if sequence:
        with st.spinner("Predicting..."):
            score, label, attn_matrix, top_residues = predict_single(
                sequence, model, device
            )

        st.success(f"### Prediction: `{label}` (score: {score:.4f})")

        # Attention heatmap
        st.markdown("### 🔍 Attention Heatmap")
        amino_acids = list(sequence.strip())  # remove whitespace/newlines

        fig, ax = plt.subplots(
            figsize=(min(20, 0.12 * len(amino_acids)), 8)
        )  # auto-resize based on sequence length
        sns.heatmap(
            attn_matrix,
            cmap="viridis",
            ax=ax,
            xticklabels=amino_acids,
            yticklabels=amino_acids,
        )
        ax.set_title("Self-Attention Matrix (Averaged Across Heads)")
        ax.set_xlabel("Residue Position")
        ax.set_ylabel("Residue Position")
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
        st.pyplot(fig)
        # Residue importance
        st.markdown("### ⭐ Top 5 Most Attended Residues")
        for idx, aa, score in top_residues:
            st.write(f"- Position {idx + 1}: `{aa}` (attention received: {score:.6f})")

    else:
        st.warning("Please enter a sequence before clicking predict.")
