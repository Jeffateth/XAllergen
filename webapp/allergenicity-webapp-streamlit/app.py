# # ==========================================
# # 🧬 Protein Allergenicity Predictor (Streamlit)
# # – Uses FAIR’s ESM-2 (1280-dim) + XGBoost
# # – Includes runtime hacks to avoid MKL/OpenMP & numpy._core issues
# # ==========================================

# # 1) ENVIRONMENT VARIABLE HACKS (must be first!)
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# os.environ["OMP_NUM_THREADS"] = "1"

# # 2) STANDARD IMPORTS
# import streamlit as st
# import torch
# import joblib
# import matplotlib.pyplot as plt

# from esm import pretrained

# # ==========================================
# # === 4) MODEL & TOKENIZER LOADING
# # ==========================================
# @st.cache_resource
# def load_esm2_model():
#     model, alphabet = pretrained.esm2_t33_650M_UR50D()
#     model.eval()
#     batch_converter = alphabet.get_batch_converter()
#     return model, batch_converter

# @st.cache_resource
# def load_xgb_model(path):
#     return joblib.load(path)

# # ==========================================
# # === 5) PREDICTION FUNCTION
# # ==========================================
# def predict(sequence, esm_model, batch_converter, xgb_model):
#     seq = ''.join(filter(str.isalpha, sequence.strip().upper()))
#     valid = set("ACDEFGHIKLMNPQRSTVWY")

#     if not seq:
#         return "⚠️ Please enter a protein sequence.", None
#     if any(aa not in valid for aa in seq):
#         return "❌ Invalid sequence: use only 20 standard amino acids.", None
#     if not (5 <= len(seq) <= 2000):
#         return "⚠️ Sequence length must be 5–2000 AA.", None

#     # Get [CLS] embedding
#     batch = [("prot", seq)]
#     _, _, tokens = batch_converter(batch)
#     with torch.no_grad():
#         out = esm_model(tokens, repr_layers=[33], return_contacts=False)
#     emb = out["representations"][33][0, 0, :].cpu().numpy().reshape(1, -1)

#     # Predict
#     prob = xgb_model.predict_proba(emb)[0, 1]
#     pred = xgb_model.predict(emb)[0]
#     label = "🟢 Allergen" if pred else "🔴 Non-Allergen"
#     result = f"**Label:** {label}  \n**P(Allergen):** `{prob:.4f}`"

#     return result, emb

# # ==========================================
# # === 6) STREAMLIT APP
# # ==========================================
# def main():
#     st.set_page_config(page_title="Protein Allergenicity Predictor", layout="centered")
#     st.title("🧬 Protein Allergenicity Predictor")
#     st.markdown("Paste one or more protein sequences (A,C,D,...,Y) — separate them with **line breaks** — and click **Predict**.")

#     seq_input = st.text_area("Protein Sequence(s)", height=200)

#     esm_model, batch_converter = load_esm2_model()
#     xgb_model = load_xgb_model("/Users/rikardpettersson/Library/Mobile Documents/com~apple~CloudDocs/Documents/ETH Chemistry Ms/Digital Chemistry/Github_repository_Allergen/XGBoost_ESM-2_1280dim_algpred2_xgboost_model.pkl")

#     if st.button("Predict Allergenicity"):
#         import pandas as pd

#         results = []
#         for idx, raw_seq in enumerate(seq_input.strip().splitlines(), 1):
#             result_text, _ = predict(raw_seq, esm_model, batch_converter, xgb_model)
#             label = "Allergen" if "🟢" in result_text else "Non-Allergen"
#             prob = float(result_text.split("`")[1])  # extract probability
#             results.append((f"Sequence {idx}", label, prob))

#         df = pd.DataFrame(results, columns=["Sequence", "Prediction", "P(Allergen)"])
#         st.dataframe(df, use_container_width=True)






# if __name__ == "__main__":
#     main()




# ==========================================
# 🧬 Protein Allergenicity Predictor (Streamlit)
# – Uses fine-tuned ESM-2 + PyTorch classifier
# – Includes runtime hacks to avoid MKL/OpenMP & numpy._core issues
# ==========================================

# 1) ENVIRONMENT VARIABLE HACKS (must be first!)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"

# 2) STANDARD IMPORTS
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from esm import pretrained
from esm_interpret import ESMModelInterpreter

# ==========================================
# === 3) MODEL ARCHITECTURE
# ==========================================
class ESMClassifier(nn.Module):
    def __init__(self, esm_model, hidden_dim, num_layers, dropout):
        super().__init__()
        self.esm = esm_model
        input_dim = self.esm.embed_dim
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, tokens):
        with torch.no_grad():
            results = self.esm(tokens, repr_layers=[6], return_contacts=False)
        embeddings = results["representations"][6]
        cls_rep = embeddings[:, 0, :]
        logits = self.classifier(cls_rep)
        return logits.view(-1)


# ==========================================
# === 4) MODEL LOADING
# ==========================================
@st.cache_resource
def load_custom_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    esm_model_name = checkpoint.get('esm_model_name', 'esm2_t6_8M_UR50D')
    if esm_model_name == 'esm2_t6_8M_UR50D':
        esm_model, alphabet = pretrained.esm2_t6_8M_UR50D()
    elif esm_model_name == 'esm2_t12_35M_UR50D':
        esm_model, alphabet = pretrained.esm2_t12_35M_UR50D()
    elif esm_model_name == 'esm2_t30_150M_UR50D':
        esm_model, alphabet = pretrained.esm2_t30_150M_UR50D()
    else:
        raise ValueError(f"Unsupported model: {esm_model_name}")

    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()

    architecture = checkpoint["model_architecture"]
    model = ESMClassifier(
        esm_model,
        architecture["hidden_dim"],
        architecture["num_layers"],
        architecture["dropout"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, batch_converter

@st.cache_resource
def load_model_for_interpretation(model_path: str) -> ESMModelInterpreter:
    return ESMModelInterpreter(model_path)

# ==========================================
# === 5) PREDICTION FUNCTION
# ==========================================
def predict(sequence, model, batch_converter):
    seq = ''.join(filter(str.isalpha, sequence.strip().upper()))
    valid = set("ACDEFGHIKLMNPQRSTVWY")

    if not seq:
        return "⚠️ Please enter a protein sequence.", None
    if any(aa not in valid for aa in seq):
        return "❌ Invalid sequence: use only 20 standard amino acids.", None
    if not (5 <= len(seq) <= 2000):
        return "⚠️ Sequence length must be 5–2000 AA.", None

    batch = [("prot", seq)]
    _, _, tokens = batch_converter(batch)
    with torch.no_grad():
        logits = model(tokens)
        prob = torch.sigmoid(logits).item()
        pred = int(prob >= 0.5)

    label = "🟢 Allergen" if pred else "🔴 Non-Allergen"
    result = f"**Label:** {label}  \n**P(Allergen):** `{prob:.4f}`"
    return result, prob

# ==========================================
# === 6) STREAMLIT APP
# ==========================================
def main():
    st.set_page_config(page_title="Protein Allergenicity Predictor", layout="centered")
    st.title("🧬 Protein Allergenicity Predictor")
    st.markdown("Paste one or more protein sequences, separated by line breaks. Click **Predict**.")

    seq_input = st.text_area("Protein Sequence(s)", height=200)

    # Load model + batch converter
    with st.spinner("Loading ESM + classifier model..."):
        model_path = "/Users/rikardpettersson/Library/Mobile Documents/com~apple~CloudDocs/Documents/ETH Chemistry Ms/Digital Chemistry/App/fine-tuned_esm2_allergen_classifier-final_version.pt"  # 🔁 <-- CHANGE THIS
        model, batch_converter = load_custom_model(model_path)

    if st.button("Predict Allergenicity"):
        with st.spinner("Predicting..."):
            results = []
            for idx, raw_seq in enumerate(seq_input.strip().splitlines(), 1):
                result_text, prob = predict(raw_seq, model, batch_converter)
                label = "Allergen" if "🟢" in result_text else "Non-Allergen"
                results.append((f"Sequence {idx}", label, prob))

        df = pd.DataFrame(results, columns=["Sequence", "Prediction", "P(Allergen)"])
        st.session_state["last_predictions"] = df  # 🔑 STORE the result
    if "last_predictions" in st.session_state:
        st.subheader("Prediction Results")
        st.dataframe(st.session_state["last_predictions"].reset_index(drop=True), use_container_width=True)


        st.markdown("---")
        st.header("🧬 Interpret Model Prediction")
        # Parse sequences from input
        sequences = [s for s in seq_input.strip().splitlines() if s.strip()]
        if sequences:
            seq_map = {f"Sequence {i+1}": seq for i, seq in enumerate(sequences)}
            selected_label = st.selectbox("Select sequence to interpret:", list(seq_map.keys()))
            selected_seq = seq_map[selected_label]


            if st.button("Run Interpretation"):
                with st.spinner("Running interpretation..."):
                    interpreter = load_model_for_interpretation(model_path)

                    # Prediction
                    st.subheader("Prediction")
                    pred = interpreter.predict(selected_seq)
                    st.markdown(f"**Prediction (P[Allergen]):** `{pred:.4f}`")

                    # Integrated Gradients Plot
                    st.subheader("Integrated Gradients Attribution")
                    attributions, amino_acids, _ = interpreter.integrated_gradients_attributions(selected_seq)
                    fig, ax = plt.subplots(figsize=(15, 5))
                    colors = ['red' if x < 0 else 'blue' for x in attributions]
                    ax.bar(range(len(attributions)), attributions, color=colors)
                    if len(amino_acids) <= 100:
                        ax.set_xticks(range(len(amino_acids)))
                        ax.set_xticklabels(amino_acids, rotation='vertical')
                    else:
                        ax.set_xticks([])
                    ax.set_title("Integrated Gradients Attribution")
                    st.pyplot(fig)

                    # Attention Heatmap
                    st.subheader("Attention Heatmap")
                    entropies, attention_matrix, valid_tokens = interpreter.visualize_attention(selected_seq)
                    fig2, ax2 = plt.subplots(figsize=(10, 8))
                    import seaborn as sns
                    sns.heatmap(attention_matrix, xticklabels=valid_tokens, yticklabels=valid_tokens, cmap="viridis", ax=ax2)
                    ax2.set_title("Attention Heatmap")
                    st.pyplot(fig2)

                    # Top Residues by Attribution
                    st.subheader("Top Influential Residues")
                    top_df = interpreter.get_top_influential_residues(selected_seq)
                    st.dataframe(top_df)

                    # Top Attention-Sending/Receiving
                    st.subheader("Top Attention-Receiving Residues")
                    _, _ = ESMModelInterpreter.plot_top_attention_residues(attention_matrix, valid_tokens, top_k=10, mode='received')
                    st.pyplot(plt.gcf())

                    st.subheader("Top Attention-Sending Residues")
                    _, _ = ESMModelInterpreter.plot_top_attention_residues(attention_matrix, valid_tokens, top_k=10, mode='sent')
                    st.pyplot(plt.gcf())

        elif seq_input.strip():
            st.warning("No valid sequences found. Please enter protein sequences using only A,C,D,...,Y.")


if __name__ == "__main__":
    main()
