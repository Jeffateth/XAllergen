import torch
import numpy as np
from model import convATTnet

AA_DICT = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
}

def encode_sequence(seq, max_len=1000):
    encoded = [0] * (1000 - len(seq))
    for a in seq:
        encoded.append(int(AA_DICT.get(a.upper(), 0)))
    return torch.tensor(encoded).unsqueeze(0).long()  # [1, 1000]

def load_model(model_path="model.pt", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = convATTnet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def predict_single(seq, model, device):
    x = encode_sequence(seq).to(device)
    seq_len = len(seq)

    with torch.no_grad():
        output, attention = model(x, return_attention=True)
        score = output.item()
        label = "allergenicity" if score > 0.5 else "non-allergenicity"

        attn_matrix = attention.squeeze(0).mean(dim=0).cpu().numpy()
        attn_matrix = attn_matrix[-seq_len:, -seq_len:]

        # NEW: compute residue importance
        importance = attn_matrix.sum(axis=0)  # attention received per residue
        topk_idx = importance.argsort()[-5:][::-1]  # top 5
        top_residues = [(i, seq[i], importance[i]) for i in topk_idx]

    return score, label, attn_matrix, top_residues

