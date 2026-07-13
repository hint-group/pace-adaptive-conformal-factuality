
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from tqdm import tqdm
import random

# Add path for local imports if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DtACI import DtACI
from factscore_utils import (
    load_data, 
    create_stream, 
    fit_distributions, 
    compute_rmd, 
    estimate_sigmoid_params
)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ProcessedData')
DATA_DIR_JSONL = os.path.join(BASE_DIR, 'data')
DATASET_JSONL = 'factscore_dataset_v3.jsonl'
METADATA_CSV = 'factscore_final.csv'

TARGET_ALPHA = 0.2
WINDOW_SIZE = 50
GAMMA_FIXED_LIST = [0.01, 0.2] 
GAMMA_MIN = 1e-9
GAMMA_MAX = 1
CALIBRATION_SIZE = 500
STREAM_LENGTH = 6000 

# Gamma Adaptation Constants
EMA_ALPHA = 0.99
GAP_SCALE = 1

# --- 4. Conformal Prediction ---

def run_cp(cal_claims, stream_entities, cal_entities, weight=0.5):
    # Prepare calibration scores
    cal_scores = np.array([c['score'] for c in cal_claims])
    
    # Prepare calibration features for RMD (In-domain)
    cal_features = np.array([e['features'] for e in cal_entities])
    
    # Prepare background features (Entire stream + Calibration)
    bg_features = np.array([e['features'] for e in stream_entities] + [e['features'] for e in cal_entities])
    
    # Fit distributions
    mu_in, Sigma_in_inv, mu_bg, Sigma_bg_inv = fit_distributions(cal_features, bg_features)
    
    # --- Auto-calibrate Sigmoid Parameters ---
    sigmoid_center, sigmoid_slope = estimate_sigmoid_params(cal_features, mu_in, Sigma_in_inv, mu_bg, Sigma_bg_inv)
    
    # --- Initialize Algorithms ---
    
    # 1. Our Method (Adaptive Gamma)
    theta_adaptive = TARGET_ALPHA
    p_rmd = 0.0
    gamma_history = []
    alpha_adaptive_history = []
    results_adaptive = []
    retention_adaptive_history = []
    
    # 2. Fixed Gamma Baselines
    theta_fixed_dict = {g: TARGET_ALPHA for g in GAMMA_FIXED_LIST}
    alpha_fixed_dict = {g: [] for g in GAMMA_FIXED_LIST}
    results_fixed_dict = {g: [] for g in GAMMA_FIXED_LIST}
    retention_fixed_dict = {g: [] for g in GAMMA_FIXED_LIST}
    
    # 3. DtACI Baseline
    gammas_dtaci = [0.001, 0.002, 0.004, 0.008, 0.0160, 0.032, 0.064, 0.128]
    model_dtaci = DtACI(gammas=gammas_dtaci, alpha=TARGET_ALPHA, eta=2.0, sigma=0.001)
    
    theta_dtaci = TARGET_ALPHA 
    alpha_dtaci_history = []
    results_dtaci = []
    retention_dtaci = []
    gamma_dtaci_history = [] 

    # 4. Standard CP Baseline 
    theta_baseline = TARGET_ALPHA
    results_baseline = []
    retention_baseline = []
    
    # Shared Logging
    rmd_history = []
    p_rmd_history = []
    ood_prob_local_history = []
    
    for t, entity in enumerate(stream_entities):
        # --- 1. RMD Calculation ---
        rmd = compute_rmd(entity['features'], mu_in, Sigma_in_inv, mu_bg, Sigma_bg_inv)
        rmd_history.append(rmd)
        
        # --- 2. DtACI Predict ---
        if t == 0:
            theta_dtaci = TARGET_ALPHA
        else:
            theta_dtaci = model_dtaci.predict()
        
        # --- 3. Thresholds Determination ---
        theta_adaptive = np.clip(theta_adaptive, 0.001, 0.999)
        theta_dtaci = np.clip(theta_dtaci, 0.001, 0.999)
        for g in GAMMA_FIXED_LIST:
            theta_fixed_dict[g] = np.clip(theta_fixed_dict[g], 0.001, 0.999)
            
        thresh_adaptive = np.quantile(cal_scores, 1 - theta_adaptive)
        thresh_dtaci = np.quantile(cal_scores, 1 - theta_dtaci)
        thresh_fixed_dict = {g: np.quantile(cal_scores, 1 - theta_fixed_dict[g]) for g in GAMMA_FIXED_LIST}
        thresh_baseline = np.quantile(cal_scores, 1 - theta_baseline)
        
        # --- 4. Prediction & Evaluation ---
        claims = entity['claims']
        scores = np.array([c['score'] for c in claims])
        ground_truths = np.array([c['is_supported'] for c in claims])
        
        def evaluate_batch(threshold, scores, ground_truths):
            accepted_mask = scores <= threshold
            n_accepted = np.sum(accepted_mask)
            if n_accepted == 0:
                fdp = 0.0
                ret = 0.0 
            else:
                false_accepted = np.sum(accepted_mask & (~ground_truths))
                fdp = false_accepted / n_accepted
                ret = n_accepted / len(scores) if len(scores) > 0 else 1.0
            return fdp, ret, accepted_mask

        # Evaluate Adaptive
        fdp_a, ret_a, mask_a = evaluate_batch(thresh_adaptive, scores, ground_truths)
        results_adaptive.append(fdp_a)
        alpha_adaptive_history.append(theta_adaptive)
        retention_adaptive_history.append(ret_a)
        
        # Evaluate DtACI
        fdp_d, ret_d, mask_d = evaluate_batch(thresh_dtaci, scores, ground_truths)
        results_dtaci.append(fdp_d)
        alpha_dtaci_history.append(theta_dtaci)
        retention_dtaci.append(ret_d)
        
        # Evaluate Fixed Gammas
        for g in GAMMA_FIXED_LIST:
            fdp_f, ret_f, _ = evaluate_batch(thresh_fixed_dict[g], scores, ground_truths)
            results_fixed_dict[g].append(fdp_f)
            alpha_fixed_dict[g].append(theta_fixed_dict[g])
            retention_fixed_dict[g].append(ret_f)

        # Evaluate Standard CP Baseline
        fdp_b, ret_b, _ = evaluate_batch(thresh_baseline, scores, ground_truths)
        results_baseline.append(fdp_b)
        retention_baseline.append(ret_b)
            
        # --- 5. Updates ---
        model_dtaci.update_fdp(fdp_d) 
        gamma_dtaci_history.append(model_dtaci.get_weighted_gamma())
        
        # Update Fixed Gammas
        for g in GAMMA_FIXED_LIST:
            theta_fixed_dict[g] -= g * (TARGET_ALPHA - results_fixed_dict[g][-1])
            
        # Update Our Adaptive Method
        ood_prob_local = 1 / (1 + np.exp(-sigmoid_slope * (rmd - sigmoid_center)))
        ood_prob_local_history.append(ood_prob_local)
        
        p_rmd = EMA_ALPHA * p_rmd + (1 - EMA_ALPHA) * ood_prob_local
        p_rmd_history.append(p_rmd)
        
        if t < WINDOW_SIZE:
            avg_err_a = TARGET_ALPHA
        else:
            avg_err_a = np.mean(results_adaptive[-WINDOW_SIZE:])
        
        gap_t = abs(avg_err_a - TARGET_ALPHA)
        p_gap = min(1.0, gap_t / GAP_SCALE)
        
        if t > 0:
            prev_p_rmd = p_rmd_history[t-1] if t-1 < len(p_rmd_history) else p_rmd
            p_t = weight * abs(p_rmd) + (1-weight) * p_gap
        else:
            p_t = weight * abs(p_rmd) + (1-weight) * p_gap

        gamma_t = np.clip(p_t, GAMMA_MIN, GAMMA_MAX)
        gamma_history.append(gamma_t)
        
        # Update Theta
        theta_adaptive -= gamma_t * (TARGET_ALPHA - fdp_a) 
        
    return {
        'error_adaptive': results_adaptive,
        'error_fixed': results_fixed_dict,  
        'error_dtaci': results_dtaci,
        'rmd': rmd_history,
        'p_rmd': p_rmd_history,
        'ood_prob_local': ood_prob_local_history,
        'gamma': gamma_history,
        'gamma_dtaci': gamma_dtaci_history,
        'alpha_adaptive': alpha_adaptive_history,
        'alpha_fixed': alpha_fixed_dict, 
        'alpha_dtaci': alpha_dtaci_history,
        'retention_adaptive': retention_adaptive_history,
        'retention_fixed': retention_fixed_dict, 
        'retention_dtaci': retention_dtaci,
        'error_baseline': results_baseline,
        'retention_baseline': retention_baseline
    }

# --- 5. Execution ---

def run_single_experiment(data, seed, weight):
    cal_claims, stream_entities, cal_entities = create_stream(data, seed, CALIBRATION_SIZE, STREAM_LENGTH)
    return run_cp(cal_claims, stream_entities, cal_entities, weight=weight)

def run_multi_trial_experiments(data, n_experiments=30):
    print(f"Running {n_experiments} experiments...")
    
    all_results = {
        'error_adaptive': [],
        'error_fixed': {g: [] for g in GAMMA_FIXED_LIST},
        'error_dtaci': [],
        'rmd': [],
        'p_rmd': [],
        'ood_prob_local': [],
        'gamma': [],
        'gamma_dtaci': [],
        'alpha_adaptive': [],
        'alpha_fixed': {g: [] for g in GAMMA_FIXED_LIST},
        'alpha_dtaci': [],
        'retention_adaptive': [],
        'retention_fixed': {g: [] for g in GAMMA_FIXED_LIST},
        'retention_dtaci': [],
        'error_baseline': [],
        'retention_baseline': []
    }
    
    for i in tqdm(range(n_experiments), desc="Running experiments"):
        res = run_single_experiment(data, seed=42 + i, weight=0.5) # DEFAULT WEIGHT 0.5 IF NOT SPECIFIED
        
        for key in ['error_adaptive', 'error_dtaci', 'rmd', 'p_rmd', 'ood_prob_local', 
                   'gamma', 'gamma_dtaci', 'alpha_adaptive', 'alpha_dtaci', 
                   'retention_adaptive', 'retention_dtaci',
                   'error_baseline', 'retention_baseline']:
            all_results[key].append(res[key])
            
        for key in ['error_fixed', 'alpha_fixed', 'retention_fixed']:
            for g in GAMMA_FIXED_LIST:
                all_results[key][g].append(res[key][g])
            
    # Convert to numpy arrays
    for key in all_results:
        if isinstance(all_results[key], dict):
            for g in GAMMA_FIXED_LIST:
                all_results[key][g] = np.array(all_results[key][g])
        else:
            all_results[key] = np.array(all_results[key])
        
    return all_results

if __name__ == "__main__":
    # Example usage
    data = load_data(DATA_DIR_JSONL, DATASET_JSONL, DATA_DIR, METADATA_CSV)
    if not data:
        print("Data load failed. Exiting.")
        exit(1)
        
    results = run_multi_trial_experiments(data, n_experiments=5) # Demo with 5
    
    from plot_utils import plot_results_multi
    plot_results_multi(results, n_experiments=5, window_size=WINDOW_SIZE, target_alpha=TARGET_ALPHA, gamma_fixed_list=GAMMA_FIXED_LIST, output_prefix="demo_run")
