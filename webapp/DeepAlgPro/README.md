# DeepAlgPro: an interpretable deep neural network model for predicting allergenic proteins
## Introduction
Allergies have become an emerging public health problem worldwide.It is critical to evaluate potential allergens, especially today when the number of modified proteins in food, therapeutic drugs, and biopharmaceuticals is increasing rapidly. Here, we proposed a software, called DeepAlgPro, that combined a convolutional neural network (CNN) with Multi-Headed Self-Attention (MHSA) and was suitable for large-scale prediction of allergens. 

## Requirements
- Platform requirement<br>
We trained the model under linux OS, but it can also be run under windows, mac OS. Your operating system must be supported by the deep learning framework and related libraries you used to use this model. For example, our model was implemented in Pytorch 1.12.1, you must check its OS compatibility list [here](https://pytorch.org/get-started/previous-versions/) to ensure that your OS (e.g., Ubuntu, Windows, macOS) is supported.
- Device requirement<br>
This model was trained on NVIDIA GeForce RTX 3090. When using it, it is supported to run under both GPU and CPU. When the GPU is not available(`torch.cuda.is_available()=False`), the model will run using the CPU.
- Packages requirement<br>
  - python 3.9<br>
  - Bio==1.5.3<br>
  - numpy==1.23.4<br>
  - pandas==1.5.0<br>
  - scikit_learn==1.2.1<br>
  - torch==1.12.1+cu116<br>
  - torchmetrics==0.9.3<br>
## Installation
1. Download DeepAlgPro
```
git clone https://github.com/chun-he-316/DeepAlgPro.git
```
2. Install required packages<br>
```
pip3 install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
## Train and Test the model
```
usage: python main.py [-h] [-i INPUTS] [--epochs N] [--lr LR] [-b N] [--mode {train,test}]
```
#### Optional arguments
```
  -h, --help                     show this help message and exit
  -i INPUTS, --inputs INPUTS
  --epochs N                     number of total epochs to run
  --lr LR, --learning-rate LR    learning rate
  -b N, --batch-size N
  --mode {train,test}
```
#### Example
```
python main.py -i data/all.train.fasta --epochs 120 --lr 0.0001 -b 72 --mode train
```
#### Output files
The training process generates a number of files. The types of files are listed below.
- train.log file: Record loss values for each batch.
- .everyepoch.valid.txt files: Record the validation results after each training epoch.
- .pt file: The model obtained by training.
- valid.log: Results of 10-fold cross-validation.
## Use DeepAlgPro to predict allergens
```
usage: python predict.py [-h] [-i INPUTS] [-b N] [-o OUTPUT]
```
#### Optional arguments
```
  -h, --help                    show this help message and exit
  -i INPUTS, --inputs INPUTS    input file
  -b N, --batch-size N
  -o OUTPUT, --output OUTPUT    output file
```
#### Input files
The input file specified by -i is a protein sequence file; each sequence has a unique id and starts with >. The input protein sequence number must be divisible by the batch size.
#### Example
```
python predict.py -i data/all.test.fasta -o allergen.predict.txt
```
#### Output files
  The default result file is `allergenic_predict.txt`, a file with tabs as spacers. You can also specify the output file with `-o`. The first column in the output file is the id of the input protein, the second column is the score between 0 and 1 predicted by the model, and the third column value is the predicted result, allergenicity or non-allergenicity.For example,
```
        protein scores  predict result
protein_1   0.9983819723129272      allergenicity
protein_2   0.999177873134613       allergenicity
protein_3   0.000125454544823       non-allergenicity
protein_4   0.9991099238395691      allergenicity
```
## Citation
He C, Ye X, Yang Y, Hu L, Si Y, Zhao X, Chen L, Fang Q, Wei Y, Wu F, Ye G. DeepAlgPro: an interpretable deep neural network model for predicting allergenic proteins. Brief Bioinform. 2023 Jul 20;24(4):bbad246. doi: 10.1093/bib/bbad246.


# 🧬 DeepAlgPro Web: Interpretable Deep Learning for Allergen Prediction

**DeepAlgPro** is an interpretable deep neural network for predicting allergenic proteins using a combination of Convolutional Neural Networks (CNNs) and Multi-Head Self-Attention (MHSA). This repository also includes a Streamlit web app for real-time prediction and visualization of model attention.

---

## 🚀 Features

- 🧠 Predicts allergenicity of protein sequences using a pretrained deep model.
- 🔍 Visualizes self-attention as a heatmap.
- 🌟 Extracts and displays the **top 5 residues** most attended by the model.
- 💻 Lightweight web interface built with Streamlit.

---

## 📦 Requirements

This project is tested on **macOS M2 Pro** with Python **3.10.13**.  
Make sure to install compatible dependencies:

```bash
# Create and activate virtual environment
pyenv install 3.10.13
pyenv virtualenv 3.10.13 deepalgpro
pyenv activate deepalgpro

# Clone the repository
git clone https://github.com/YOUR_USERNAME/DeepAlgPro.git
cd DeepAlgPro

# Install dependencies
pip install -r requirements.txt
```
---
## 📂 File Structure

```
DeepAlgPro/
│
├── model.py                  # CNN + Attention model
├── model.pt                  # Pretrained model weights
├── predict_with_attention.py# Prediction logic w/ attention
├── app.py                    # Streamlit interface
├── data/                     # Sample FASTA files
├── requirements.txt
└── README.md
```

---

## ✅ How to Use

### 1. Run the Web App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

### 2. Paste Protein Sequence

Use 1-letter amino acid codes. Max length: 1000 characters.

```
MKTFLVALVLLSAAGITASAAVNASYSARYKTTGQGTWHDNGNSQRDQIYQQQGGRIFVKEDGAVSFSK...
```

---

### 3. Output Includes

- ✅ Allergenicity prediction (e.g. `allergenic` / `non-allergenic`)
- 🔥 Self-attention heatmap
- ⭐ Top 5 most attended residues (position, amino acid, attention score)

---

## 🧠 Model Architecture

- `Conv1D` layer to extract local features.
- `MaxPooling` and `Dropout` for regularization.
- `Multi-Head Self-Attention` to learn interpretable long-range dependencies.
- Fully-connected output with `sigmoid` for binary classification.

---

## ✨ Example Output

```
Prediction: allergenicity (score: 0.87)

Top 5 Most Attended Residues:
- Position 63: G (attention: 0.130084)
- Position 120: R (attention: 0.130047)
- ...
```

---

## 📜 Citation

> He C, et al. *DeepAlgPro: an interpretable deep neural network model for predicting allergenic proteins*. Brief Bioinform. 2023; 24(4):bbad246. [DOI](https://doi.org/10.1093/bib/bbad246)

---

## 🛠️ Future Work

- Add batch FASTA upload and CSV export
- Highlight top residues on heatmap
- Connect attention scores with known functional motifs

---

With contributions from the open-source and bioinformatics community 🧪
