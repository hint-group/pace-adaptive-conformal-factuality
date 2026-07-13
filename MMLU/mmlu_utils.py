
import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from scipy.special import softmax

def load_background_data():
    """Loads background data (e.g. WikiText) and computes embeddings."""
    possible_dirs = ['.', 'MMLU_v6_sensitivity', 'MMLU_v8_score']
    base_dir = '.'
    for d in possible_dirs:
        if os.path.exists(os.path.join(d, "background_wiki.json")):
            base_dir = d
            break
            
    params_path = os.path.join(base_dir, "background_wiki.json")
    
    if not os.path.exists(params_path):
        print(f"Warning: {params_path} not found. Returning None.")
        return None

    print(f"Loading background data from {params_path}...")
    with open(params_path, 'r') as f:
        texts = json.load(f)
    
    if not texts:
        return None
        
    print(f"Encoding {len(texts)} background documents...")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    
    embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=True, device=DEVICE)
    return embeddings

def calculate_score(row):
    """
    Calculates the non-conformity score (1 - probability of the correct answer).
    Assumes row has 'logits_options' and 'answer'.
    """
    answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    logits = row['logits_options']
    probs = softmax(logits)
    correct_idx = answer_map.get(row['answer'])
    if correct_idx is None or correct_idx >= len(probs):
        return None # Handle edge cases
    
    prob_correct = probs[correct_idx]
    # CP Score: non-conformity. Higher prob -> Lower score
    return 1.0 - prob_correct

def get_rolling(arr, w):
    """
    Computes rolling mean of an array.
    """
    return pd.Series(arr).rolling(window=w, min_periods=1).mean().values

def safe_rmse(arr, target, indices):
    """
    Computes RMSE for a subset of indices safely.
    """
    if len(indices) == 0: return 0.0
    return np.sqrt(np.mean((arr[indices] - target)**2))

def get_recovery_time(err_series, jump_start, target_alpha, tolerance=0.01, k=50):
    """
    Calculates recovery time.
    """
    lower_bound = target_alpha - tolerance
    upper_bound = target_alpha + tolerance
    
    if jump_start >= len(err_series):
        return None
    
    for t in range(jump_start, len(err_series) - k):
        window = err_series[t : t+k]
        if np.all((window >= lower_bound) & (window <= upper_bound)):
            return t - jump_start
    return None

def compute_rmd(Z_stream, mu_in, Sigma_in_inv, mu_bg, Sigma_bg_inv):
    """
    Computes Relative Mahalanobis Distance (RMD).
    """
    diff_in = Z_stream - mu_in
    md_in = np.sum(diff_in @ Sigma_in_inv * diff_in, axis=1)
    
    diff_bg = Z_stream - mu_bg
    md_bg = np.sum(diff_bg @ Sigma_bg_inv * diff_bg, axis=1)
    
    return md_in - md_bg

def get_q_hat_function(calibration_scores):
    """
    Returns a function hat_Q(q_level) that computes quantile on calibration_scores.
    """
    def hat_Q(q_level):
        if q_level <= 0: return -np.inf
        if q_level >= 1: return np.inf
        return np.quantile(calibration_scores, q_level, method='higher')
    return hat_Q
