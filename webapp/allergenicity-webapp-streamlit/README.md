# XAllergen Web User Interface and Additional Information

This repository provides a web interface for XAllergen: an explainable machine learning tool for protein allergenicity prediction using ESM-2 transformer models. The interface enables users to input protein sequences, visualize results, and explore model explainability—all in their web browser.

## Features

- **Protein Allergenicity Prediction**  
  Upload or paste protein sequences and receive allergenicity predictions powered by fine-tuned ESM-2 models.

- **Interactive Visualizations**  
  Explore attention-based explainability, integrated gradients, model barplots, and 3D structure visualizations.

- **User-Friendly Web Interface**  
  No need for command-line interaction—everything runs in your web browser using [Streamlit](https://streamlit.io/).

---

## Getting Started
``bash
1. Clone the Repository


git clone https://github.com/Jeffateth/XAllergen.git
cd XAllergen/webapp/allergenicity-webapp-streamlit

2. Set Up the Environment
It is recommended to use conda or venv to create an isolated Python environment.
Required Python version: Python 3.9 or 3.10

# Using conda (recommended)
conda create -n xallergen_env python=3.10
conda activate xallergen_env

# Install requirements
pip install -r requirements.txt

or 


3. Download Model Weights
The app requires pre-trained or fine-tuned model weights. Download or place the weights in the specified folder (see documentation or the app’s sidebar for details).

4. Run the Web Application
streamlit run app.py


Input your protein sequence in the provided text box, or upload a file.


Troubleshooting

If you experience errors with model loading, ensure that the weights are downloaded and placed in the correct directory.
For issues with package dependencies, check your Python version and consider using a fresh environment.
If you encounter Streamlit or Torch errors, see the FAQ section in the repository.
Additional Information

Further documentation, screenshots, and technical details can be found in the project’s GitHub repository.

If you have questions or issues, please open an issue on GitHub.
