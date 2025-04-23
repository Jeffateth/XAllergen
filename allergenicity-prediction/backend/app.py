# app.py
import os
import json
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Load tokenizer and model
MODEL_PATH = 'model.safetensors'
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", cache_dir="./cache")

# Load special tokens if available
try:
    with open('special_tokens_map.json', 'r') as f:
        special_tokens = json.load(f)
        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)
except FileNotFoundError:
    print("No special tokens map found. Using default tokenizer.")

# Load the fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=None,
    config=config,
    state_dict=torch.load(MODEL_PATH, map_location=torch.device('cpu'))
)
model.eval()

# Load vocabulary if available
try:
    with open('vocab.txt', 'r') as f:
        vocab = f.read().splitlines()
except FileNotFoundError:
    vocab = ["Non-allergenic", "Allergenic"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'sequence' not in data:
        return jsonify({"error": "No protein sequence provided"}), 400
    
    sequence = data['sequence']
    # Basic input validation
    if not sequence or len(sequence) < 5:
        return jsonify({"error": "Sequence too short"}), 400
    
    # Check if sequence contains valid amino acids
    valid_aa = "ACDEFGHIKLMNPQRSTVWY"
    if not all(aa in valid_aa for aa in sequence.upper()):
        return jsonify({"error": "Sequence contains invalid amino acids"}), 400
    
    try:
        # Tokenize sequence
        inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Get class label
        if len(vocab) > prediction:
            label = vocab[prediction]
        else:
            label = "Class " + str(prediction)
        
        return jsonify({
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "allergenic_probability": round(probabilities[0][1].item() * 100, 2) if probabilities.shape[1] > 1 else None
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/', methods=['GET'])
def home():
    return """
    <html>
        <head><title>ESM-2 Allergenicity Prediction API</title></head>
        <body>
            <h1>ESM-2 Allergenicity Prediction API</h1>
            <p>This API provides allergenicity predictions for protein sequences using a fine-tuned ESM-2 model.</p>
            <p>Use POST /predict endpoint with a JSON payload containing your protein sequence.</p>
        </body>
    </html>
    """

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)