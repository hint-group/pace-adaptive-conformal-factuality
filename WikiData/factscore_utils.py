
import os
import json
import random
import numpy as np
import pandas as pd
import scipy.stats as st

def load_data(data_dir_jsonl, dataset_jsonl, data_dir, metadata_csv):
    """
    Loads and preprocesses the FactScore dataset and metadata.
    
    Returns:
        list: List of processed entity data dictionaries.
    """
    print("Loading data...")
    jsonl_path = os.path.join(data_dir_jsonl, dataset_jsonl)
    print(f"Reading dataset from {jsonl_path}...")
    
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found.")
        return []

    with open(jsonl_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    print(f"Loading auxiliary files from {data_dir}...")
    metadata_path = os.path.join(data_dir, metadata_csv)
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return []
        
    metadata_df = pd.read_csv(metadata_path)
    
    # Metadata map
    valid_entities = {}
    for _, row in metadata_df.iterrows():
        valid_entities[row['Name']] = row['Views']
        
    processed_data = []
    count_found = 0
    count_missing_metadata = 0
    
    for item in dataset:
        prompt = item.get('prompt')
        entity_name = item.get('entity_name')
        
        if not prompt or not entity_name:
            continue
            
        if entity_name not in valid_entities:
            count_missing_metadata += 1
            continue
            
        views = valid_entities[entity_name]
        claims = item.get('claims', [])
        
        entity_claims = []
        for claim in claims:
            # claim structure: {"message": "...", "scores": {"frequency": ..., "self_eval": ...}, "label": 1/0}
            scores = claim.get('scores', {})
            freq = scores.get('frequency', 0.0)
            self_eval = scores.get('self_eval', 0.0)
            label = claim.get('label')
            
            is_supported = bool(label)
            
            claim_data = {
                'entity': entity_name,
                'is_supported': is_supported,
                'frequency': freq,
                'self_eval': self_eval, 
                'page_views': views,
                'score': 1.0 / (freq + self_eval + 1e-9) # Non-conformity score
            }
            entity_claims.append(claim_data)
            
        if entity_claims:
            # Calculate entity-level features for RMD
            avg_self_eval = np.mean([c['self_eval'] for c in entity_claims])
            avg_freq = np.mean([c['frequency'] for c in entity_claims])
            
            # RMD features: [avg_self_eval, avg_freq]
            processed_data.append({
                'entity': entity_name,
                'claims': entity_claims,
                'features': np.array([avg_self_eval, avg_freq]),
                'page_views': views
            })
            count_found += 1
            
    print(f"Processed {count_found} entities.")
    print(f"Skipped: {count_missing_metadata} missing metadata.")
    return processed_data

def create_stream(data, seed=42, calibration_size=500, stream_length=6000):
    # Sort by page views (Ascending)
    def get_views(x):
        try:
            val = x['page_views']
            if isinstance(val, str):
                val = val.replace(',', '').strip()
            f = float(val)
            if np.isnan(f): return -1.0
            return f
        except (ValueError, TypeError):
            return -1.0
            
    data.sort(key=get_views)
    n = len(data)
    
    # Define Tiers (20% buckets)
    # Tier 1 (Top 20%): 80-100% -> ID
    idx_75 = int(n * 0.75)
    idx_50 = int(n * 0.5)
    idx_25 = int(n * 0.25)
    
    tier1_id = data[idx_75:]
    tier2_ood = data[idx_50:idx_75]
    tier3_ood = data[idx_25:idx_50]
    tier4_ood = data[:idx_25]
    
    print(f"Tier 1 (ID, >{tier1_id[0]['page_views']}): {len(tier1_id)}")
    
    # Calibration Set (From ID)
    random.seed(seed)
    cal_size = min(calibration_size, len(tier1_id))
    calibration_entities = random.sample(tier1_id, cal_size)
    
    # Build Calibration Set of Claims
    cal_claims = []
    for ent in calibration_entities:
        cal_claims.extend(ent['claims'])
        
    # Construct Stream
    # Pattern: 800 ID -> 200 OOD
    STABLE_LEN = 800
    JUMP_LEN = 200
    NUM_CYCLES = 6
    
    ood_tiers = [tier2_ood, tier3_ood, tier4_ood]
    
    stream_entities = []
    time_segments = [] 
    current_time = 0
    
    for i in range(NUM_CYCLES):
        # 1. Stable Segment (ID)
        stable_segment = random.choices(tier1_id, k=STABLE_LEN)
        stream_entities.extend(stable_segment)
        
        time_segments.append((current_time, current_time + STABLE_LEN, "ID", "Tier 1"))
        current_time += STABLE_LEN
        
        # 2. Jump Segment (OOD)
        tier_idx = i % len(ood_tiers)
        selected_ood_tier = ood_tiers[tier_idx]
        tier_name = f"Tier {tier_idx + 2}"
        
        jump_segment = random.choices(selected_ood_tier, k=JUMP_LEN)
        stream_entities.extend(jump_segment)
        
        time_segments.append((current_time, current_time + JUMP_LEN, "OOD", tier_name))
        current_time += JUMP_LEN
    
    stream_entities = stream_entities[:stream_length]
    
    print("\nSHIFT DISTRIBUTION TABLE")
    print("="*60)
    print(f"{'Time':<15} {'Length':<8} {'Type':<10} {'Subject/Tier':<15}")
    print("-"*60)
    for start, end, stype, sub in time_segments:
        print(f"[{start}, {end})      {end-start:<8} {stype:<10} {sub:<15}")
    print("="*60 + "\n")

    return cal_claims, stream_entities, calibration_entities

def fit_distributions(cal_features, bg_features):
    """
    Fits Gaussian distributions to in-domain (calibration) and background features.
    """
    # 1. In-domain
    mu_in = np.mean(cal_features, axis=0)
    Sigma_in = np.cov(cal_features.T, ddof=1)
    
    # 2. Background
    mu_bg = np.mean(bg_features, axis=0)
    Sigma_bg = np.cov(bg_features.T, ddof=1)
    
    # Ensure 2D
    if Sigma_in.ndim == 0: Sigma_in = Sigma_in.reshape(1, 1)
    if Sigma_bg.ndim == 0: Sigma_bg = Sigma_bg.reshape(1, 1)
    
    # 3. Regularization & Inverse
    REGULARIZATION_LAMBDA = 1e-6
    I = np.eye(Sigma_in.shape[0])
    
    Sigma_in_reg = Sigma_in + REGULARIZATION_LAMBDA * I
    Sigma_bg_reg = Sigma_bg + REGULARIZATION_LAMBDA * I
    
    try:
        Sigma_in_inv = np.linalg.inv(Sigma_in_reg)
    except np.linalg.LinAlgError:
        Sigma_in_inv = np.linalg.pinv(Sigma_in_reg)
        
    try:
        Sigma_bg_inv = np.linalg.inv(Sigma_bg_reg)
    except np.linalg.LinAlgError:
        Sigma_bg_inv = np.linalg.pinv(Sigma_bg_reg)
        
    return mu_in, Sigma_in_inv, mu_bg, Sigma_bg_inv

def compute_rmd(features, mu_in, Sigma_in_inv, mu_bg, Sigma_bg_inv):
    """
    Computes Relative Mahalanobis Distance (RMD).
    """
    diff_in = features - mu_in
    md_in = diff_in.T @ Sigma_in_inv @ diff_in
    
    diff_bg = features - mu_bg
    md_bg = diff_bg.T @ Sigma_bg_inv @ diff_bg
    
    return md_in - md_bg

def estimate_sigmoid_params(cal_features, mu_in, Sigma_in_inv, mu_bg, Sigma_bg_inv, quantile=0.8):
    """
    Estimates sigmoid parameters based on calibration RMDs.
    """
    cal_rmds = []
    for feat in cal_features:
        r = compute_rmd(feat, mu_in, Sigma_in_inv, mu_bg, Sigma_bg_inv)
        cal_rmds.append(r)
    cal_rmds = np.array(cal_rmds)
    
    sigmoid_center = np.quantile(cal_rmds, quantile) 
    sigmoid_slope = 10.0 / sigmoid_center if sigmoid_center != 0 else 1.0
    
    print(f"Auto-calibrated Sigmoid: Center={sigmoid_center:.4f}, Slope={sigmoid_slope:.4f}")
    return sigmoid_center, sigmoid_slope
