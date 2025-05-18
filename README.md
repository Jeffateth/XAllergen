<<<<<<< HEAD
# 🧬 XAllergen: Deep Learning for Allergenicity Prediction

**XAllergen** is a collaborative course project in digital chemistry at ETH Zurich focused on using AI to predict protein allergenicity from sequence and structure. By combining traditional bioinformatics descriptors, protein language model embeddings, and structural data, we aim to build an end-to-end pipeline for accurate and interpretable allergenicity prediction. We aim to develop a novel approach to integrate 3D protein data to our model. 

Our google drive link for larger files: https://drive.google.com/drive/folders/1Jjc4-SqccRb75_gBKfQ-pPC6kVCk8WeY?usp=sharing
=======
# XAllergen

**(eXplainable Allergenicity Prediction of Proteins using AI)**
>>>>>>> c66dc97e7134c85cd5ba3038f3a689ebe148a7e3

---

## 🧪 Project Overview

<<<<<<< HEAD
The goal of this project is to:

- Predict whether a protein is **allergenic** or **non-allergenic**
- Integrate **sequence-based**, **structural**, and **embedding-based** features
- Evaluate various models, from **Random Forests**, and **CNNs**
- Explore input formats: **full protein sequences**
- Assess performance with robust metrics: **ROC-AUC**, **F1**, **Accuracy**, **MCC**, and **Standard Error**
=======
**XAllergen** is a predictive and interpretable tool designed to assess the allergenicity of proteins based solely on their amino acid sequences. We fine-tuned the ESM-2 protein language model and trained neural architectures with integrated attention to classify sequences. Additionally, we implemented interpretability features (e.g., Integrated Gradients + 3D visualization) and deployed the system through a user-friendly web interface.
>>>>>>> c66dc97e7134c85cd5ba3038f3a689ebe148a7e3

---

## 🧬 Key Features

* ⚙️ **Protein Embedding**: Fine-tuned [ESM-2](https://github.com/facebookresearch/esm) models for deep representation of protein sequences.
* 🧠 **Models**: XGBoost, Ridge Regression, FFNN, 1D-CNN, and full ESM-2 fine-tuning.
* 🎯 **Evaluation**: Accuracy, F1-score, MCC, AUC-ROC.
* 🎨 **Interpretability**: Integrated Gradients and 3D attribution visualization (PyMOL and py3Dmol).
* 🌐 **Web App**: Streamlit interface for interactive protein input and real-time visual predictions.


<<<<<<< HEAD
### 🔍 What Happens When You Input a New Protein Sequence?

1. **The sequence is validated**: only standard amino acids (A, C, D, ..., Y) are accepted.
2. **The model tokenizes and evaluates** the sequence using a fine-tuned ESM-2 transformer.
3. **Raw model probabilities** are calibrated using `IsotonicRegression` for realism.
4. The result is compared against a threshold (e.g., 0.5) to label the sequence:
   - 🟢 Allergen
   - 🔴 Non-allergen
5. **Optional**: SHAP explainability highlights the influence of specific amino acids.
=======
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

>>>>>>> c66dc97e7134c85cd5ba3038f3a689ebe148a7e3

This system allows researchers to **quickly assess allergenicity risk** from sequence alone — ideal for filtering, triaging, and guiding lab work. However, it does **not replace experimental validation**.
