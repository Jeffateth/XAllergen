Great! Here's an updated version of the `README.md` that includes the **Google Drive link** for downloading large files (e.g., trained model checkpoints, full structure predictions, high-resolution figures):

---

# XAllergen

**(eXplainable Allergenicity Prediction of Proteins using AI)**

## 🌟 One-Sentence Summary

An explainable AI model that predicts protein allergenicity from amino acid sequences, accessible via an interactive Streamlit web app.

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

## 📊 Model Performance (Test Set)

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.85  |
| Precision | 0.84  |
| Recall    | 0.86  |
| F1-score  | 0.85  |
| ROC-AUC   | 0.92  |

---

## 🧠 Explainable AI

* **Integrated Gradients**: Highlights influential amino acids.
* **3D Attribution Mapping**: Visual overlays of attributions on protein structures using PyMOL.


