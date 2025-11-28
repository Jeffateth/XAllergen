# XAllergen

**(eXplainable Allergenicity Prediction of Proteins using AI)**

<p align="center">
  <a href="docs/XAllergen_final_documentation.pdf">
    <img src="https://img.shields.io/badge/📄_Documentation-Full_Report-b31b1b?style=for-the-badge" alt="Documentation">
  </a>
  <a href="docs/XAllergen_poster.pdf">
    <img src="https://img.shields.io/badge/🎨_Poster-ETH_Poster_Presentation-1f77b4?style=for-the-badge" alt="Poster">
  </a>
  <a href="https://xallergen.streamlit.app">
    <img src="https://img.shields.io/badge/🌐_Demo-Live_Application-00ab41?style=for-the-badge" alt="Live Demo">
  </a>
  <a href="https://drive.google.com/drive/folders/1Jjc4-SqccRb75_gBKfQ-pPC6kVCk8WeY?usp=sharing">
    <img src="https://img.shields.io/badge/💾_Downloads-More_Data-f4b400?style=for-the-badge" alt="Downloads">
  </a>
</p>

> ⚠️ **Demo Notice:** Hosted on Streamlit's free tier. Initial load may take ~8 minutes if the app is sleeping.

---

## 🧪 Project Overview

**XAllergen** is a predictive and interpretable tool designed to assess the allergenicity of proteins based solely on their amino acid sequences. We fine-tuned the ESM-2 protein language model and trained neural architectures with integrated attention to classify sequences. Additionally, we implemented interpretability features (e.g., Integrated Gradients + 3D visualization) and deployed the system through a user-friendly web interface.

---

## 🧬 Key Features

* ⚙️ **Protein Embedding**: Fine-tuned [ESM-2](https://github.com/facebookresearch/esm) models for deep representation of protein sequences.
* 🧠 **Models**: XGBoost, Ridge Regression, FFNN, 1D-CNN, and full ESM-2 fine-tuning.
* 🎯 **Evaluation**: Accuracy, F1-score, MCC, AUC-ROC, Precision, Sensitivity, Specificity.
* 🎨 **Interpretability**: Integrated Gradients and 3D attribution visualization (py3Dmol).
* 🌐 **Web App**: Streamlit interface for interactive protein input and real-time visual predictions.

---

## 📁 Project Structure

```plaintext
ALLERGENPREDICT/
├── data/                        # Datasets (AlgPred 2.0, IEDB, amino acid properties)
├── models/                     # Fine-tuned model weights
├── webapp/                     # Streamlit web interface
├── src/                        # Training, analysis and preprocessing code
├── results/                    # Model evaluation and reports
```

---

## 🚀 Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/Jeffateth/XAllergen.git
cd XAllergen
```

### 2. Set Up Environment

#### Using conda
```bash
conda create -n XAllergen python=3.11
conda activate XAllergen
```
#### Using venv
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```
### 3. Install Packages

```bash
cd webapp/allergenicity-webapp-streamlit
pip install -r requirements.txt
```

### 4. Launch the Web App

```bash
streamlit run app.py
```

---

## 🔗 Downloads

Large files such as full model weights, PDB files, and high-resolution visualizations are hosted on Google Drive:

📁 **[Download from Google Drive](https://drive.google.com/drive/folders/1Jjc4-SqccRb75_gBKfQ-pPC6kVCk8WeY?usp=sharing)**
