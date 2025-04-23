from flask import Flask, request, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load tokenizer and model from local `model/` directory
# Ensure the directory contains: config.json, model.safetensors, vocab.txt, special_tokens_map.json
MODEL_DIR = "model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Read and sanitize the protein sequence
        seq = request.form['sequence'].strip().upper().replace(' ', '')

        # Tokenize the input sequence
        inputs = tokenizer(seq, return_tensors='pt', padding=True, truncation=True)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits            # shape: (batch_size=1, num_labels=2)
            probs = torch.softmax(logits, dim=1)
            allergen_prob = probs[0, 1].item()  # probability for class index 1 (Allergen)

        # Prepare results for template
        result = {
            'prediction': 'Allergen' if allergen_prob > 0.5 else 'Non-allergen',
            'probability': f"{allergen_prob:.4f}"  
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    # Use 0.0.0.0 to accept external connections (e.g., Docker)
    app.run(host='0.0.0.0', port=8000)