# 🧬 XAllergen: Deep Learning for Allergenicity Prediction

**XAllergen** is a collaborative course project in digital chemistry at ETH Zurich focused on using AI to predict protein allergenicity from sequence and structure. By combining traditional bioinformatics descriptors, protein language model embeddings, and structural data, we aim to build an end-to-end pipeline for accurate and interpretable allergenicity prediction. We aim to develop a novel approach to integrate 3D protein data to our model. 

Our google drive link for larger files: https://drive.google.com/drive/folders/1Jjc4-SqccRb75_gBKfQ-pPC6kVCk8WeY?usp=sharing

---

## 🚀 Project Overview

The goal of this project is to:

- Predict whether a protein is **allergenic** or **non-allergenic**
- Integrate **sequence-based**, **structural**, and **embedding-based** features
- Evaluate various models, from **Random Forests**, and **CNNs**
- Explore input formats: **full protein sequences**
- Assess performance with robust metrics: **ROC-AUC**, **F1**, **Accuracy**, **MCC**, and **Standard Error**

---

## ✅ Current Progress

### Weekly Highlights

#### 🔹 27.03.2025
- Curated high-quality negative dataset  
- Implemented a baseline **Random Forest + AAC descriptor** model (AUC = 0.62)  
- Explored evaluation metrics: **F1**, **Accuracy**, **AUC**  
- Investigated advanced descriptors: **AAIndex**, **Propy3**  
- Discussed integration of **ESM-2 embeddings**

#### 🔹 03.04.2025
- Analyzed training datasets from **AlgPred 2.0**  
- Evaluated model on a **2024 benchmark dataset** (92.6% accuracy)  
- Successfully deployed **ESMFold** to generate 3D PDB structures from sequences  
- Planned integration of **3D structural features** via **DSSP**


### 🔍 What Happens When You Input a New Protein Sequence?

1. **The sequence is validated**: only standard amino acids (A, C, D, ..., Y) are accepted.
2. **The model tokenizes and evaluates** the sequence using a fine-tuned ESM-2 transformer.
3. **Raw model probabilities** are calibrated using `IsotonicRegression` for realism.
4. The result is compared against a threshold (e.g., 0.5) to label the sequence:
   - 🟢 Allergen
   - 🔴 Non-allergen
5. **Optional**: SHAP explainability highlights the influence of specific amino acids.

This system allows researchers to **quickly assess allergenicity risk** from sequence alone — ideal for filtering, triaging, and guiding lab work. However, it does **not replace experimental validation**.
