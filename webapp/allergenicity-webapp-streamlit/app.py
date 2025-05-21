# ==========================================
# XAllergen (Streamlit)
# – Uses fine-tuned ESM-2 +Integrated Gradients
# ==========================================

# Import necessary libraries
import os
import streamlit as st
import py3Dmol
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from esm import pretrained
import streamlit.components.v1 as components
import plotly.graph_objs as go
import sys
import logging

# Add the current directory to path to find modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import the ESM interpreter
from esm_interpret import ESMModelInterpreter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# === 0) CONFIGURATION
# ==========================================

# Use a relative path for the model - will look in models/Integrated_gradient directory
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "Integrated_gradient"
)
DEFAULT_MODEL_NAME = "fine-tuned_esm2_allergen_classifier-final_version.pt"
MODEL_PATH = os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME)


# Directory for PDB files
PDB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdb_files")
DEFAULT_PDB = "P_7275.pdb"
PDB_PATH = os.path.join(PDB_DIR, DEFAULT_PDB)

# ==========================================
# === 1) FIX FOR LIBRARY CRASH AND CPU OVERLOAD
# ==========================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"


# ==========================================
# === 2) MODEL ARCHITECTURE
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
# === 3) MODEL LOADING
# ==========================================
@st.cache_resource
def load_custom_model(checkpoint_path):
    try:
        # Check if model exists
        if not os.path.exists(checkpoint_path):
            st.error(f"Model file not found at: {checkpoint_path}")
            st.info("Please download the model and place it in the 'models' directory.")
            return None, None

        # Load the checkpoint with weights_only=False (be cautious)
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=False
        )

        esm_model_name = checkpoint.get("esm_model_name", "esm2_t6_8M_UR50D")
        if esm_model_name == "esm2_t6_8M_UR50D":
            esm_model, alphabet = pretrained.esm2_t6_8M_UR50D()
        elif esm_model_name == "esm2_t12_35M_UR50D":
            esm_model, alphabet = pretrained.esm2_t12_35M_UR50D()
        elif esm_model_name == "esm2_t30_150M_UR50D":
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
            architecture["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, batch_converter
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logger.error(f"Error loading model: {str(e)}")
        return None, None


@st.cache_resource
def load_model_for_interpretation(model_path: str) -> ESMModelInterpreter:
    try:
        return ESMModelInterpreter(model_path)
    except Exception as e:
        st.error(f"Error loading interpreter: {str(e)}")
        logger.error(f"Error loading interpreter: {str(e)}")
        return None


# ==========================================
# === 4) PREDICTION FUNCTION
# ==========================================
def predict(sequence, model, batch_converter):
    """
    Predict allergenicity for a single protein sequence
    """
    if model is None or batch_converter is None:
        return "⚠️ Model not loaded correctly. Please check the model file.", None

    seq = "".join(filter(str.isalpha, sequence.strip().upper()))
    valid = set("ACDEFGHIKLMNPQRSTVWY")

    if not seq:
        return "⚠️ Please enter a protein sequence.", None
    if any(aa not in valid for aa in seq):
        return "❌ Invalid sequence: use only 20 standard amino acids.", None
    if not (5 <= len(seq) <= 1022):
        return "⚠️ Sequence length must be 5–1022 AA.", None

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
# === 5) 3D STRUCTURE VISUALIZATION
# ==========================================
def get_color(attr, norm_score):
    """
    Get a color for visualization based on attribution score
    """
    if norm_score < 1e-6:
        return "#FFFFFF"  # White for near-zero attributions
    elif attr > 0:
        # Blue → White fade
        r = int(255 * (1 - norm_score))
        g = int(255 * (1 - norm_score))
        b = 255
    else:
        # Red → White fade
        r = 255
        g = int(255 * (1 - norm_score))
        b = int(255 * (1 - norm_score))
    return f"#{r:02x}{g:02x}{b:02x}"


def show_3d_structure(pdb_path, attributions, sequence, spin=True):
    """
    Create a 3D visualization of the protein structure colored by attributions
    """
    try:
        with open(pdb_path) as f:
            pdb_data = f.read()

        abs_attr = np.abs(attributions)
        norm_attr = abs_attr / abs_attr.max() if abs_attr.max() > 0 else abs_attr

        view = py3Dmol.view(width=800, height=500)
        view.addModel(pdb_data, "pdb")
        view.setStyle({"cartoon": {"opacity": 1.0}})

        for i, (aa, score) in enumerate(zip(sequence, norm_attr)):
            color = get_color(attributions[i], score)
            resi = i + 1
            view.addStyle({"resi": str(resi)}, {"cartoon": {"color": color}})

        view.zoomTo()
        view.spin(spin)
        return view
    except Exception as e:
        st.error(f"Error creating 3D structure: {str(e)}")
        logger.error(f"Error creating 3D structure: {str(e)}")
        return None


def render_pdb_in_streamlit(view):
    """
    Render the 3D structure viewer in Streamlit
    """
    if view is None:
        st.error("Unable to create 3D visualization. PDB file may be missing.")
        return

    inner_html = view._make_html()

    bordered_viewer = f"""
    <div style="
        width: 800px;
        height: 500px;
        border: 1px solid black;
        overflow: hidden;
        background-color: rgba(255, 255, 255, 0.2);
    ">
        {inner_html}
    </div>
    """

    color_legend_horizontal = """
    <div style="width: 600px; display: flex; flex-direction: column; align-items: center; margin-top: 14px;">
        <!-- Color bar -->
        <div style="width: 500px; height: 24px; background: linear-gradient(to right, red, white, blue); 
                    border: 1px solid black;">
        </div>

        <!-- Tick labels -->
        <div style="width: 500px; display: flex; justify-content: space-between;
                    font-family: monospace; font-size: 14px; margin-top: 4px;">
            <span>-1</span>
            <span>-0.5</span>
            <span>0</span>
            <span>0.5</span>
            <span>1</span>
        </div>

        <!-- Caption -->
        <div style="margin-top: 8px; font-family: monospace; font-size: 14px; font-weight: bold;">
            Integrated Gradients Attribution
        </div>
    </div>
    """

    combined = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
        {bordered_viewer}
        {color_legend_horizontal}
    </div>
    """

    st.components.v1.html(combined, height=640, width=820)


# ==========================================
# === 6) STREAMLIT APP UI
# ==========================================
def main():
    """
    Main function for the Streamlit app
    """
    global MODEL_PATH, PDB_PATH  # Add this line
    st.set_page_config(page_title="Protein Allergenicity Predictor", layout="centered")
    st.title("🧬 Protein Allergenicity Predictor")

    # Show readme link
    st.markdown(
        "📚 For instructions, please refer to the [README.md](https://github.com/Jeffateth/XAllergen/blob/main/README.md) file."
    )

    # Check for model and PDB directories
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        st.warning(
            f"Created models directory at {MODEL_DIR}. Please place model file here."
        )

    if not os.path.exists(PDB_DIR):
        os.makedirs(PDB_DIR)
        st.warning(f"Created PDB directory at {PDB_DIR}. Please place PDB files here.")

    # Allow users to select a different model if multiple models are available
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
    if model_files:
        selected_model = st.sidebar.selectbox(
            "Select model",
            model_files,
            index=(
                model_files.index(DEFAULT_MODEL_NAME)
                if DEFAULT_MODEL_NAME in model_files
                else 0
            ),
        )
        MODEL_PATH = os.path.join(MODEL_DIR, selected_model)
    else:
        st.sidebar.warning("No model files found in models directory.")

    # Check if PDB files are available
    pdb_files = [f for f in os.listdir(PDB_DIR) if f.endswith(".pdb")]
    if pdb_files:
        selected_pdb = st.sidebar.selectbox(
            "Select PDB for 3D visualization",
            pdb_files,
            index=pdb_files.index(DEFAULT_PDB) if DEFAULT_PDB in pdb_files else 0,
        )
        PDB_PATH = os.path.join(PDB_DIR, selected_pdb)
    else:
        st.sidebar.warning(
            "No PDB files found in pdb_files directory. 3D visualization will be disabled."
        )

    st.markdown(
        "Paste one or more protein sequences, separated by line breaks. Click **Predict**."
    )

    EXAMPLE_SEQUENCE = "MKNYLSFGMFALLFALTFGTVNSVQAIAGPEWLLDRPSVNNSQLVVSVAGTVEGTNQDISLKFFEIDLTSRPAHGGKTEQGLSPKSKPFATDSGAMSHKLEKADLLKAIQEQLIANVHSNDDYFEVIDFASDATITDRNGKVYFADKDGSVTLPTQPVQEFLLSGHVRVRPYKEKPIQNQAKSVDVEYTVQFTPLNPDDDFRPGLKDTKLLKTLAIGDTITSQELLAQAQSILNKNHPGYTIYERDSSIVTHDNDIFRTILPMDQEFTYRVKNREQAYRINKKSGLNEEINNTDLISEKYYVLKKGEKPYDPFDRSHLKLFTIKYVDVDTNELLKSEQLLTASERNLDFRDLYDPRDKAKLLYNNLDAFGIMDYTLTGKVEDNHDDTNRIITVYMGKRPEGENASYHLAYDKDRYTEEEREVYSYLRYTGTPIPDNPNDK"

    if "seq_input" not in st.session_state:
        st.session_state["seq_input"] = ""

    if st.button("Insert Example Sequence"):
        st.session_state["seq_input"] = EXAMPLE_SEQUENCE

    seq_input = st.text_area(
        "Protein Sequence(s)",
        height=200,
        value=st.session_state["seq_input"],
        key="seq_input_area",
    )

    # Load model + batch converter
    with st.spinner("Loading ESM + classifier model..."):
        model, batch_converter = load_custom_model(MODEL_PATH)
        if model is None:
            st.error(f"Failed to load model from {MODEL_PATH}.")
            st.info(
                "Please download the model file and place it in the 'models' directory."
            )
            st.stop()

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
        st.dataframe(
            st.session_state["last_predictions"],
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("---")
        st.header("🧬 Interpret Model Prediction")
        # Parse sequences from input
        sequences = [s for s in seq_input.strip().splitlines() if s.strip()]
        if sequences:
            seq_map = {f"Sequence {i+1}": seq for i, seq in enumerate(sequences)}
            selected_label = st.selectbox(
                "Select sequence to interpret:", list(seq_map.keys())
            )
            selected_seq = seq_map[selected_label]

            if st.button("Run Interpretation"):
                with st.spinner("Running interpretation..."):
                    interpreter = load_model_for_interpretation(MODEL_PATH)
                    if interpreter is None:
                        st.error("Failed to initialize model interpreter.")
                        st.stop()

                    pred = interpreter.predict(selected_seq)
                    st.session_state["interpretation_done"] = True
                    st.session_state["selected_seq"] = selected_seq
                    st.session_state["pred"] = pred
                    attributions, amino_acids, _ = (
                        interpreter.integrated_gradients_attributions(selected_seq)
                    )
                    st.session_state["attributions"] = attributions
                    st.session_state["amino_acids"] = amino_acids

                    # Check if PDB file exists
                    if os.path.exists(PDB_PATH):
                        st.session_state["pdb_path"] = PDB_PATH
                    else:
                        st.warning(
                            f"PDB file not found at {PDB_PATH}. 3D visualization will be disabled."
                        )
                        st.session_state["pdb_path"] = None

                    st.session_state["top_df"] = (
                        interpreter.get_top_influential_residues(selected_seq)
                    )
                    entropies, attention_matrix, valid_tokens = (
                        interpreter.visualize_attention(selected_seq)
                    )
                    st.session_state["attention_matrix"] = attention_matrix
                    st.session_state["valid_tokens"] = valid_tokens
                    st.session_state["spin"] = True
        st.session_state.setdefault("show_interpretation_outputs", False)

        if st.session_state.get("interpretation_done"):
            st.subheader("Prediction")

            prediction_label = (
                "Allergen" if st.session_state["pred"] >= 0.5 else "Non-Allergen"
            )
            prediction_df = pd.DataFrame(
                [[prediction_label, st.session_state["pred"]]],
                columns=["Prediction", "P(Allergen)"],
            )
            st.dataframe(prediction_df, use_container_width=True, hide_index=True)

            st.subheader("Integrated Gradients Attribution")

            attributions = st.session_state["attributions"]
            amino_acids = st.session_state["amino_acids"]
            colors = ["red" if x < 0 else "blue" for x in attributions]

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(range(len(attributions))),
                        y=attributions,
                        marker_color=colors,
                        text=amino_acids if len(amino_acids) <= 100 else None,
                        hovertext=[
                            f"{aa}: {score:.4f}"
                            for aa, score in zip(amino_acids, attributions)
                        ],
                        hoverinfo="text",
                    )
                ]
            )

            fig.update_layout(
                xaxis=dict(
                    title="Residue Position",
                    tickmode="array",
                    tickvals=list(range(len(amino_acids))),
                    ticktext=amino_acids if len(amino_acids) <= 100 else None,
                ),
                yaxis=dict(title="Attribution Score"),
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Top Influential Residues")
            df = st.session_state["top_df"]
            st.dataframe(df.reset_index(drop=True), hide_index=True)

            st.subheader("Attention Heatmap")
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                st.session_state["attention_matrix"],
                xticklabels=st.session_state["valid_tokens"],
                yticklabels=st.session_state["valid_tokens"],
                cmap="viridis",
                ax=ax2,
            )
            ax2.set_title("Attention Heatmap")
            st.pyplot(fig2)

            st.subheader("Top Attention-Receiving Residues")
            _, _ = ESMModelInterpreter.plot_top_attention_residues(
                st.session_state["attention_matrix"],
                st.session_state["valid_tokens"],
                top_k=10,
                mode="received",
            )
            st.pyplot(plt.gcf())

        if (
            st.session_state.get("interpretation_done")
            and st.session_state.get("pdb_path")
            and os.path.exists(st.session_state.get("pdb_path", ""))
        ):
            st.markdown("---")
            st.subheader("🧬 3D Structure Visualization")

            # --- Toggle Spin ---
            spin_now = st.toggle(
                "Spin structure",
                value=st.session_state.get("spin", False),
                key="spin_toggle",
            )

            # Check if value changed and trigger rerun
            if "spin" in st.session_state and st.session_state["spin"] != spin_now:
                st.session_state["spin"] = spin_now
                st.rerun()  # Ensures immediate visual/state update

            view = show_3d_structure(
                st.session_state["pdb_path"],
                st.session_state["attributions"],
                st.session_state["amino_acids"],
                spin=st.session_state["spin"],
            )
            render_pdb_in_streamlit(view)


if __name__ == "__main__":
    main()
