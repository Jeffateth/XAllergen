import streamlit as st
import torch
import numpy as np
import joblib
from esm import pretrained
import os

# === Load ESM2 model ===
@st.cache_resource
def load_esm2_model():
    model, alphabet = pretrained.esm2_t6_8M_UR50D()  # adjust if you used another variant
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter

# === Load trained bootstrap models ===
@st.cache_resource
def load_bootstrap_models(model_dir, base_model_name, n_models=5):
    models = []
    for i in range(1, n_models + 1):
        model_path = os.path.join(model_dir, f"{base_model_name}_bootstrap_{i}.pkl")
        models.append(joblib.load(model_path))
    return models

# === Prediction Function using bootstrap models ===
def predict_with_uncertainty(sequence, esm_model, batch_converter, bootstrap_models, alphabet):
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

    # Get prediction probability from all bootstrap models
    probs = []
    preds = []
    for model in bootstrap_models:
        prob = model.predict_proba(cls_embedding)[0][1]
        pred = model.predict(cls_embedding)[0]
        probs.append(prob)
        preds.append(pred)

    mean_prob = np.mean(probs)
    std_prob = np.std(probs)
    majority_vote = int(np.round(np.mean(preds)))  # majority 0 or 1

    return majority_vote, mean_prob, std_prob

# === Streamlit App ===
def main():
    st.set_page_config(page_title="Protein Allergenicity Predictor", layout="centered")
    st.title("🧬 Protein Allergenicity Predictor")
    st.markdown("Enter a protein sequence below and click **Predict** to determine if it's an allergen.\n"
                "_(Prediction shows mean ± std deviation across 5 bootstrap models)_")

    # Input
    sequence_input = st.text_area("Protein Sequence", height=150)

    # Load models
    esm_model, alphabet, batch_converter = load_esm2_model()
    model_name = "XGBoost"
    transformer_name = "ESM-2_320dim"
    dataset_name = "algpred2"
    model_base_name = f"{model_name}_{transformer_name}_{dataset_name}"
    bootstrap_models = load_bootstrap_models(".", model_base_name, n_models=5)  # Assuming models in same folder

    # Predict
    if st.button("🔍 Predict Allergenicity"):
        if sequence_input.strip() == "":
            st.warning("Please enter a protein sequence.")
        else:
            pred_class, mean_prob, std_prob = predict_with_uncertainty(sequence_input, esm_model, batch_converter, bootstrap_models, alphabet)

            # Make a nice result box
            result_html = f"""
            <div style="border: 2px solid {'#e53935' if pred_class == 1 else '#2e7d32'};
                        padding: 16px; border-radius: 10px; background-color: #f9f9f9;">
                <h4 style="margin: 0; color: {'#e53935' if pred_class == 1 else '#2e7d32'};">
                    {'🧪 Allergen Detected' if pred_class == 1 else '✅ Non-Allergen'}
                </h4>
                <p style="margin: 5px 0 0 0;">Probability: <b>{mean_prob:.4f} ± {std_prob:.4f}</b></p>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
