# XAllergen

**(eXplainable Allergenicity Prediction of Proteins using AI)**

---

## 🧪 Project Overview

**XAllergen** is a predictive and interpretable tool designed to assess the allergenicity of proteins based solely on their amino acid sequences. We fine-tuned the ESM-2 protein language model and trained neural architectures with integrated attention to classify sequences. Additionally, we implemented interpretability features (e.g., Integrated Gradients + 3D visualization) and deployed the system through a user-friendly web interface.

---

## 🧬 Key Features

* ⚙️ **Protein Embedding**: Fine-tuned [ESM-2](https://github.com/facebookresearch/esm) models for deep representation of protein sequences.
* 🧠 **Models**: XGBoost, Ridge Regression, FFNN, 1D-CNN, and full ESM-2 fine-tuning.
* 🎯 **Evaluation**: Accuracy, F1-score, MCC, AUC-ROC.
* 🎨 **Interpretability**: Integrated Gradients and 3D attribution visualization (PyMOL and py3Dmol).
* 🌐 **Web App**: Streamlit interface for interactive protein input and real-time visual predictions.

---

## 📁 Project Structure

```plaintext
ALLERGENPREDICT/
├── data/                        # Datasets (AlgPred 2.0, IEDB, amino acid properties)
├── models/                     # Fine-tuned model weights
├── Integrated_gradient/        # IG maps and 3D attribution results
├── webapp/                     # Streamlit web interface
├── src/                        # Training, analysis and preprocessing code
├── results/                    # Model evaluation and reports
├── requirements.txt            # Dependencies
├── env.yaml                    # Conda environment definition
├── *.ipynb / *.py              # Analysis and training notebooks/scripts
```

---

## 🚀 Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/XAllergen.git
cd XAllergen
```

### 2. Set Up Environment

```bash
conda env create -f env.yaml
conda activate xallergen
```

### 3. Launch the Web App

```bash
cd webapp/allergenicity-webapp-streamlit
streamlit run app.py
```

---

## 🔗 Downloads

Large files such as full model weights, PDB files, and high-resolution visualizations are hosted on Google Drive:

📁 **[Download from Google Drive](https://drive.google.com/drive/folders/1Jjc4-SqccRb75_gBKfQ-pPC6kVCk8WeY?usp=sharing)**

---

## 🧠 Explainable AI

* **Integrated Gradients**: Highlights influential amino acids.
* **3D Attribution Mapping**: Visual overlays of attributions on protein structures using PyMOL.


