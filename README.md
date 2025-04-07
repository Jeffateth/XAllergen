# 🧬 AllergenAI: Machine Learning for Allergenicity Prediction

**AllergenAI** is a collaborative course project in digital chemistry at ETH Zurich focused on using AI to predict protein allergenicity from sequence and structure. By combining traditional bioinformatics descriptors, protein language model embeddings, and structural data, we aim to build an end-to-end pipeline for accurate and interpretable allergenicity prediction. We aim to develop a novel approach to integrate 3D protein data to our model. 

Our google drive link for larger files: https://drive.google.com/drive/folders/1Jjc4-SqccRb75_gBKfQ-pPC6kVCk8WeY?usp=sharing

---

## 🚀 Project Overview

The goal of this project is to:

- Predict whether a protein is **allergenic** or **non-allergenic**
- Integrate **sequence-based**, **structural**, and **embedding-based** features
- Evaluate various models, from **Random Forests** to **GNNs** and **CNNs**
- Explore input formats: **epitope sequences** vs. **full protein sequences**
- Assess performance with robust metrics: **AUC**, **F1**, **Accuracy**, and **Standard Error**

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

---

## 🧠 Technical Vision: Sequence-to-Structure-to-Prediction

We are working toward an advanced architecture where:

1. **FASTA sequences** from AlgPred 2.0 are converted into **3D PDB structures** using **ESMFold**, a faster alternative to AlphaFold.
2. **DSSP** is used to extract structural features (e.g., secondary structure, solvent accessibility).
3. Features are integrated into our existing data table containing:
   - Full **ESM-2 embeddings**
   - **Protein IDs**
   - **Labels** (allergenic/non-allergenic)
4. The complete feature matrix is used to train downstream models:
   - **CNNs** (Convolutional Neural Networks)
   - **GNNs** (Graph Neural Networks), especially for 3D structure-based input

### ⚙️ Challenges & Infrastructure

- ESMFold ran successfully on a **MacBook M2 Pro** (~20 min per 500-aa protein), but this is not scalable for our **20,000-sequence dataset**
- **Google Colab Free** crashes due to memory limits (12.7 GB RAM, 16 GB T4 GPU), even when processing one sequence at a time
- Potential solutions:
  - **Google Colab Pro+** (with A100 GPU)
  - **Access to ETH GPU workstations or Euler cluster with GPU support**

 ## Current Model Architecture:

 AlgPred 2.0 Data Source
│
├── ① FASTA protein sequence
│     └──▶ ESM-2 Embedding
│
├── ② PDB 3D protein structure (predicted via ESMFold)
│     └──▶ DSSP-extracted structural features
│
├── ③ Epitope sequence (potentially)
│
└────────────┬───────────────────────────────
             │
             ▼
      [Combined Feature Input]
             │
             ▼
         CNN / GNN / XGBoost
             │
             ▼
   Allergenicity Prediction



