import re
from flask import Flask, request, render_template, jsonify
import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager, login_required

app = Flask(__name__)
app.secret_key = 'replace_with_secure_key'

# Rate limiting
limiter = Limiter(app, key_func=get_remote_address, default_limits=["100/day", "10/minute"])

# User authentication (placeholder)
login_manager = LoginManager(app)
# ... configure user loader, login view ...

# Load model
def load_model():
    model_dir = "model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# SHAP explainer
explainer = shap.Explainer(model, tokenizer)

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    result = None
    explanation = None
    error = None
    if request.method == 'POST':
        seq = request.form['sequence'].strip().upper().replace(' ', '')

        # Input validation: non-empty, only standard amino acids
        if not seq:
            error = "Please enter an amino acid sequence."
        elif not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq):
            error = "Invalid amino acid sequence. Use only A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y."
        else:
            thresh = float(request.form.get('threshold', 0.5))
            inputs = tokenizer(seq, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
                allergen_prob = probs[0,1].item()
            shap_vals = explainer([seq])[0].values
            result = {
                'prediction': 'Allergen' if allergen_prob > thresh else 'Non-allergen',
                'probability': allergen_prob,
                'threshold': thresh
            }
            explanation = shap_vals.tolist()

    return render_template(
        'index.html',
        result=result,
        explanation=explanation,
        error=error
    )

@app.route('/api/predict', methods=['POST'])
@limiter.limit("20/minute")
def api_predict():
    data = request.get_json(force=True)
    seq = data.get('sequence', '').strip().upper().replace(' ', '')
    # Input validation for API
    if not seq or not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq):
        return jsonify({'error': 'Invalid amino acid sequence.'}), 400
    inputs = tokenizer(seq, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        allergen_prob = probs[0,1].item()
    return jsonify({
        'prediction': 'Allergen' if allergen_prob>0.5 else 'Non-allergen',
        'probability': round(allergen_prob,4)
    })