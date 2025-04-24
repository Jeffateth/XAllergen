import streamlit as st
import torch
import numpy as np
import joblib
from esm import pretrained

# === Load ESM2 model ===
@st.cache_resource
def load_esm2_model():
    model, alphabet = pretrained.esm2_t6_8M_UR50D()  # adjust if you used another variant
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter

# === Load trained XGBoost model ===
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# === Prediction Function ===
def predict(sequence, esm_model, batch_converter, clf, alphabet):
    sequence = sequence.strip().upper()
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if not all(aa in valid_aa for aa in sequence):
        return "❌ Invalid sequence: only standard amino acids (20) are allowed."

    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
    token_representations = results["representations"][6]
    cls_embedding = token_representations[0, 0, :].numpy().reshape(1, -1)

    pred = clf.predict(cls_embedding)[0]
    prob = clf.predict_proba(cls_embedding)[0][1]

    return f"Prediction: **{'Allergen' if pred == 1 else 'Non-Allergen'}**  \nProbability: `{prob:.4f}`"

# === Streamlit App ===
def main():
    st.set_page_config(page_title="Protein Allergenicity Predictor", layout="centered")
    st.title("Protein Allergenicity Predictor")
    st.markdown("Enter a protein sequence below and click **Predict** to determine if it's an allergen.")

    # Input
    sequence_input = st.text_area("Protein Sequence", height=150)

    # Load models
    esm_model, alphabet, batch_converter = load_esm2_model()
    model_name = "XGBoost"
    transformer_name = "ESM-2_320dim"
    dataset_name = "algpred2"
    model_path = f"{model_name}_{transformer_name}_{dataset_name}_xgboost_model.pkl"
    clf = load_model(model_path)

    # Predict
    if st.button("Predict Allergenicity"):
        if sequence_input.strip() == "":
            st.warning("Please enter a protein sequence.")
        else:
            result = predict(sequence_input, esm_model, batch_converter, clf, alphabet)
            st.markdown(result)

if __name__ == "__main__":
    main()
