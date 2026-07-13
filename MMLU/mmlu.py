
import os
import json
import pickle
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from sentence_transformers import SentenceTransformer

# Import Helpers
try:
    from mmlu_utils import (
        load_background_data, 
        calculate_score, 
        compute_rmd, 
        get_q_hat_function
    )
    from DtACI import DtACI
except ImportError:
    # Handle running from root
    import sys
    sys.path.append(os.path.join(os.getcwd(), 'MMLU_v8_score')) # Assuming generic path structure
    from mmlu_utils import (
        load_background_data, 
        calculate_score, 
        compute_rmd, 
        get_q_hat_function
    )
    from DtACI import DtACI

# ==========================================
# Configuration & Constants
# ==========================================
T = 6000
TARGET_ALPHA = 0.1
GAMMA_FIXED_LIST = [0.01, 0.10]
GAMMA_MIN = 1e-9
GAMMA_MAX = 0.20
WINDOW_SIZE = 50

STABLE_SEGMENT_SIZE = 800
JUMP_SEGMENT_SIZE = 200
cycle_len = STABLE_SEGMENT_SIZE + JUMP_SEGMENT_SIZE
NUM_CYCLES = int(np.ceil(T / cycle_len))

# Define JUMPS for later analysis/plotting
JUMPS = { (i * cycle_len + STABLE_SEGMENT_SIZE) : JUMP_SEGMENT_SIZE for i in range(NUM_CYCLES) }

# ==========================================
# 1. Validation Data Loading & Preprocessing
# ==========================================
print("\n--- Loading Data ---")
json_path = "mmlu_10k.json"
pkl_path = "Qwen-72B_mmlu_10k_base_icl1.pkl"

if not os.path.exists(json_path) or not os.path.exists(pkl_path):
    # Try looking in MMLU dir if running from root
    if os.path.exists(os.path.join("MMLU", json_path)):
        json_path = os.path.join("MMLU", json_path)
        pkl_path = os.path.join("MMLU", pkl_path)
    else:
        print(f"Error: {json_path} or {pkl_path} not found.")
        # We allow import execution even if file missing (for logic check), but run will fail
        # exit(1) 

# Load JSON & PKL if exist
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        mmlu_data = json.load(f)
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)

    # Convert to DataFrames and Merge
    df_json = pd.DataFrame(mmlu_data)
    df_pkl = pd.DataFrame(pkl_data)
    df_merged = pd.merge(df_json, df_pkl, on='id', how='inner')
    print(f"Merged Data Shape: {df_merged.shape}")

    # Calculate Scores
    print("Calculating scores...")
    df_merged['score'] = df_merged.apply(calculate_score, axis=1)
    df_merged['is_ground_truth'] = True 
    df_merged.rename(columns={'id': 'question_id'}, inplace=True)
    df_merged['question_id'] = df_merged['question_id'].astype(str)

    df_scores = df_merged.dropna(subset=['score']).copy()
    print(f"Valid Scores Loaded: {len(df_scores)}")

    # --- Embeddings ---
    print("\n--- Computing Embeddings ---")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

    unique_questions = df_scores[['question_id', 'question']].drop_duplicates()
    question_texts = unique_questions['question'].tolist()
    question_ids = unique_questions['question_id'].tolist()

    print(f"Encoding {len(question_texts)} unique questions...")
    emb_cache_path = "mmlu_embeddings_cache.pkl"
    if os.path.exists(emb_cache_path):
        print("Loading embeddings from cache...")
        with open(emb_cache_path, 'rb') as f:
            embedding_map = pickle.load(f)
    else:
        question_embeddings = embed_model.encode(question_texts, batch_size=32, show_progress_bar=True, device=DEVICE)
        embedding_map = {qid: emb for qid, emb in zip(question_ids, question_embeddings)}
        with open(emb_cache_path, 'wb') as f:
            pickle.dump(embedding_map, f)

    df_scores['embedding'] = df_scores['question_id'].map(embedding_map)
    print("Embeddings added.")

    # --- Background Data ---
    bg_embeddings = load_background_data()
    if bg_embeddings is None:
        print("Warning: Background data not found. RMD might degrade.")
        
    # ==========================================
    # 2. Setup Calibration Pools (Global)
    # ==========================================
    all_subjects = sorted(df_scores['subcategory'].unique())
    CALIBRATION_SUBJECTS = ['business']
    TEST_STREAM_SUBJECTS = [s for s in all_subjects if s not in CALIBRATION_SUBJECTS]

    # Calibration Pool Filter (Actual split is per-trial)
    df_cal_pool = df_scores[df_scores['subcategory'].isin(CALIBRATION_SUBJECTS)]
    cal_all_qids = df_cal_pool['question_id'].unique()

    print(f"\nCalibration Subject Pool Size: {len(cal_all_qids)} questions")

else:
    # Placeholder if files don't exist (e.g. strict separate environment check)
    df_scores = None
    cal_all_qids = []
    bg_embeddings = None
    TEST_STREAM_SUBJECTS = []

# ==========================================
# 3. Experiment Logic
# ==========================================
def run_single_experiment_mmlu(seed, weight=0.5):
    random.seed(seed)
    np.random.seed(seed)
    
    if df_scores is None: return {} # Guard
    
    # 1. Split Calibration / ID Test
    cal_qids, test_id_qids = train_test_split(cal_all_qids, test_size=0.5, random_state=seed)
    
    # 2. Calibration Set & Q_hat
    df_cal = df_scores[df_scores['question_id'].isin(cal_qids)]
    calibration_scores = df_cal[df_cal['is_ground_truth'] == True]['score'].values
    
    hat_Q = get_q_hat_function(calibration_scores)
    
    # 3. RMD Stat Calculation
    Z_in = np.stack(df_cal['embedding'].values)
    mu_in = np.mean(Z_in, axis=0)
    Sigma_in = np.cov(Z_in.T, ddof=1)
    
    if bg_embeddings is not None and len(bg_embeddings) > 0:
        Z_bg = bg_embeddings
        mu_bg = np.mean(Z_bg, axis=0)
        Sigma_bg = np.cov(Z_bg.T, ddof=1)
    else:
        mu_bg = mu_in
        Sigma_bg = Sigma_in
        
    try:
        Sigma_in_inv = np.linalg.inv(Sigma_in)
        Sigma_bg_inv = np.linalg.inv(Sigma_bg)
    except np.linalg.LinAlgError:
        Sigma_in_inv = np.linalg.inv(Sigma_in + 1e-5 * np.eye(len(mu_in)))
        Sigma_bg_inv = np.linalg.inv(Sigma_bg + 1e-5 * np.eye(len(mu_bg)))
        
    # 4. Prepare Stream Pools
    choice_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    if 'ground_truth_label' not in df_scores.columns:
        df_scores['ground_truth_label'] = df_scores['answer'].map(choice_map)
    
    df_test_id_data = df_scores[df_scores['question_id'].isin(test_id_qids)].copy()
    pool_id = df_test_id_data[['question_id', 'ground_truth_label', 'embedding', 'score']]
    
    df_test_ood_data = df_scores[df_scores['subject'].isin(TEST_STREAM_SUBJECTS)].copy()
    pool_ood_by_subject = {}
    for subject in TEST_STREAM_SUBJECTS:
        subject_data = df_test_ood_data[df_test_ood_data['subject'] == subject]
        if len(subject_data) > 0:
            pool_ood_by_subject[subject] = subject_data[['question_id', 'ground_truth_label', 'embedding', 'score']]
            
    # 5. Construct Stream
    test_stream_list = []
    ood_subjects_list = list(pool_ood_by_subject.keys())
    random.shuffle(ood_subjects_list)
    current_time = 0
    
    for i in range(NUM_CYCLES):
        # Stable
        current_stable_len = min(STABLE_SEGMENT_SIZE, T - current_time)
        if current_stable_len <= 0: break
        
        stable_segment = pool_id.sample(n=current_stable_len, replace=True).copy()
        stable_segment['segment_type'] = 'stable'
        test_stream_list.append(stable_segment)
        current_time += current_stable_len
        
        # Jump
        current_jump_len = min(JUMP_SEGMENT_SIZE, T - current_time)
        if current_jump_len <= 0: break
        
        selected_subject = ood_subjects_list[i % len(ood_subjects_list)]
        pool_ood_jump = pool_ood_by_subject[selected_subject]
        
        jump_segment = pool_ood_jump.sample(n=current_jump_len, replace=True).copy()
        jump_segment['segment_type'] = 'jump'
        test_stream_list.append(jump_segment)
        current_time += current_jump_len
        
    test_stream_df = pd.concat(test_stream_list).reset_index(drop=True).iloc[:T]
    
    # Process RMD
    Z_stream = np.stack(test_stream_df['embedding'].values)
    rmd_raw = compute_rmd(Z_stream, mu_in, Sigma_in_inv, mu_bg, Sigma_bg_inv)
    
    # Auto-calibrate Sigmoid 
    cal_rmds = compute_rmd(Z_in, mu_in, Sigma_in_inv, mu_bg, Sigma_bg_inv)

    SIGMOID_CENTER = np.quantile(cal_rmds, 0.8) 
    SIGMOID_SLOPE = 10.0 / SIGMOID_CENTER if SIGMOID_CENTER != 0 else 1.0
    
    print(f"Auto-calibrated Sigmoid: Center={SIGMOID_CENTER:.4f}, Slope={SIGMOID_SLOPE:.4f}")
    ood_prob_local = 1 / (1 + np.exp(-SIGMOID_SLOPE * (rmd_raw - SIGMOID_CENTER)))
    
    # 6. CP Methods Initialization
    alpha_aci_ours = np.zeros(T)
    alpha_aci_ours[0] = TARGET_ALPHA
    err_aci_ours = []
    size_aci_ours = []
    
    alpha_aci_fixed = {g: np.zeros(T) for g in GAMMA_FIXED_LIST}
    for g in GAMMA_FIXED_LIST: alpha_aci_fixed[g][0] = TARGET_ALPHA
    err_aci_fixed = {g: [] for g in GAMMA_FIXED_LIST}
    size_aci_fixed = {g: [] for g in GAMMA_FIXED_LIST}
    
    alpha_baseline = np.zeros(T)
    err_baseline = []
    size_baseline = []
    
    alpha_dtaci = np.zeros(T)
    alpha_dtaci[0] = TARGET_ALPHA
    err_dtaci = []
    size_dtaci = []
    model_dtaci = DtACI(gammas=[0.001, 0.002, 0.004, 0.008, 0.0160, 0.032, 0.064, 0.128], alpha=TARGET_ALPHA)
    
    p_rmd = ood_prob_local[0]
    p_rmd_history = []
    p_gap_history = []
    p_t_history = []
    gamma_t_history = []
    gamma_dtaci_history = []
    
    EMA_ALPHA = 0.9
    
    test_stream_complete = test_stream_df.merge(
        df_scores[['question_id', 'logits_options', 'question', 'choices']], 
        on='question_id', 
        how='left'
    )
    
    example_log = []
    log_indices = [100, 799, 805, 850, 950, 1805, 2805] 
    
    # Loop
    for t in range(T):
        row = test_stream_complete.iloc[t]
        logits = row['logits_options']
        probs = softmax(logits)
        scores_all = 1.0 - probs
        correct_idx = int(row['ground_truth_label'])
        
        # Baseline
        alpha_baseline[t] = TARGET_ALPHA
        q = hat_Q(1.0 - alpha_baseline[t])
        set_indices = np.where(scores_all <= q)[0]
        is_error = 1.0 if correct_idx not in set_indices else 0.0
        err_baseline.append(is_error)
        size_baseline.append(len(set_indices))
        
        # Fixed
        for g in GAMMA_FIXED_LIST:
            alpha_aci_fixed[g][t] = np.clip(alpha_aci_fixed[g][t], 0.0, 1.0)
            q = hat_Q(1.0 - alpha_aci_fixed[g][t])
            set_indices = np.where(scores_all <= q)[0]
            is_error = 1.0 if correct_idx not in set_indices else 0.0
            err_aci_fixed[g].append(is_error)
            size_aci_fixed[g].append(len(set_indices))
            
            if t < T-1:
                alpha_aci_fixed[g][t+1] = alpha_aci_fixed[g][t] + g * (TARGET_ALPHA - is_error)
                alpha_aci_fixed[g][t+1] = np.clip(alpha_aci_fixed[g][t+1], 0.0, 1.0)
                
        # Ours
        alpha_aci_ours[t] = np.clip(alpha_aci_ours[t], 0.0, 1.0)
        q = hat_Q(1.0 - alpha_aci_ours[t])
        set_indices = np.where(scores_all <= q)[0]
        is_error_ours = 1.0 if correct_idx not in set_indices else 0.0
        err_aci_ours.append(is_error_ours)
        size_aci_ours.append(len(set_indices))
        
        # Update logic
        p_rmd = EMA_ALPHA * p_rmd + (1 - EMA_ALPHA) * ood_prob_local[t]
        
        if t < WINDOW_SIZE:
            avg_err = TARGET_ALPHA
        else:
            avg_err = np.mean(err_aci_ours[t-WINDOW_SIZE+1 : t+1])
        gap = abs(avg_err - TARGET_ALPHA)
        p_gap = min(1.0, gap)
        
        if t > 0:
            prev_p_rmd = p_rmd_history[t-1] if t-1 < len(p_rmd_history) else p_rmd
            p_t = weight * abs(p_rmd - prev_p_rmd) + (1-weight) * p_gap
        else:
            p_t = weight * p_rmd + (1-weight) * p_gap
            
        gamma_t = p_t
        
        gamma_t_history.append(gamma_t)
        p_t_history.append(p_t)
        p_gap_history.append(p_gap)
        p_rmd_history.append(p_rmd)
        
        if t < T-1:
            alpha_aci_ours[t+1] = alpha_aci_ours[t] + gamma_t * (TARGET_ALPHA - is_error_ours)
            alpha_aci_ours[t+1] = np.clip(alpha_aci_ours[t+1], 0.0, 1.0)
            
        # DtACI
        if t > 0:
            alpha_dtaci[t] = model_dtaci.predict()
        q = hat_Q(1.0 - alpha_dtaci[t])
        set_indices = np.where(scores_all <= q)[0]
        is_error = 1.0 if correct_idx not in set_indices else 0.0
        err_dtaci.append(is_error)
        size_dtaci.append(len(set_indices))
        
        # Beta_t calc
        y_t = scores_all[correct_idx]
        cdf_val = np.mean(calibration_scores <= y_t)
        beta_t = 1.0 - cdf_val
        beta_t = np.clip(beta_t, 1e-6, 1-1e-6)
        model_dtaci.update(beta_t)
        gamma_dtaci_history.append(model_dtaci.get_weighted_gamma())
        
        # Capture Example
        if t in log_indices:
            idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
            def indices_to_letters(indices):
                return [idx_to_letter[i] for i in indices]

            question_text = row['question']
            choices_dict = row['choices'] 
            
            q_base = hat_Q(1.0 - alpha_baseline[t])
            set_base_idx = np.where(scores_all <= q_base)[0]
            
            q_ours_val = hat_Q(1.0 - alpha_aci_ours[t])
            set_ours_idx = np.where(scores_all <= q_ours_val)[0]
            
            q_dtaci_val = hat_Q(1.0 - alpha_dtaci[t])
            set_dtaci_idx = np.where(scores_all <= q_dtaci_val)[0]
            
            g_ex = 0.01
            q_fixed = hat_Q(1.0 - alpha_aci_fixed[g_ex][t])
            set_fixed_idx = np.where(scores_all <= q_fixed)[0]

            ex_entry = {
                't': t,
                'segment_type': row['segment_type'],
                'question': question_text,
                'choices': choices_dict,
                'ground_truth': f"{idx_to_letter[correct_idx]}: {choices_dict.get(idx_to_letter[correct_idx], '')}",
                'probs': {idx_to_letter[i]: float(probs[i]) for i in range(len(probs))},
                'sets': {
                    'Baseline': indices_to_letters(set_base_idx),
                    'Ours': indices_to_letters(set_ours_idx),
                    'DtACI': indices_to_letters(set_dtaci_idx),
                    'Fixed_0.01': indices_to_letters(set_fixed_idx)
                },
                'alphas': {
                    'Baseline': float(alpha_baseline[t]),
                    'Ours': float(alpha_aci_ours[t]),
                    'DtACI': float(alpha_dtaci[t])
                }
            }
            example_log.append(ex_entry)
        
    return {
        'alpha_aci_ours': alpha_aci_ours,
        'err_aci_ours': np.array(err_aci_ours),
        'size_aci_ours': np.array(size_aci_ours),
        'err_baseline': np.array(err_baseline),
        'size_baseline': np.array(size_baseline),
        'err_dtaci': np.array(err_dtaci),
        'size_dtaci': np.array(size_dtaci),
        'alpha_dtaci': alpha_dtaci,
        'gamma_t_history': np.array(gamma_t_history),
        'p_rmd_history': np.array(p_rmd_history),
        'p_gap_history': np.array(p_gap_history),
        'gamma_dtaci_history': np.array(gamma_dtaci_history),
        'alpha_baseline': alpha_baseline,
        'alpha_aci_fixed': alpha_aci_fixed,
        'err_aci_fixed': err_aci_fixed,
        'size_aci_fixed': size_aci_fixed,
        'example_log': example_log,
        'md_in': 0, # Placeholder if needed
        'ood_prob_local': ood_prob_local
    }

# ==========================================
# 4. Multi-Trial Run Wrapper
# ==========================================
if __name__ == "__main__":
    print(f"\nRunning 20 experiments (demo)...") 

    agg_results = {
        'err_aci_ours': [], 'err_baseline': [], 'err_dtaci': [],
        'size_aci_ours': [], 'size_baseline': [], 'size_dtaci': [],
        'alpha_aci_ours': [], 'alpha_baseline': [], 'alpha_dtaci': [],
        'gamma_t': [], 'gamma_dtaci': [],
        'p_rmd': [], 'p_gap': [],
        'err_aci_fixed': {g: [] for g in GAMMA_FIXED_LIST},
        'alpha_aci_fixed': {g: [] for g in GAMMA_FIXED_LIST},
        'size_aci_fixed': {g: [] for g in GAMMA_FIXED_LIST}
    }

    for i in range(20):
        print(f"Trial {i+1}...")
        res = run_single_experiment_mmlu(seed=42+i)
        if not res: continue

        agg_results['err_aci_ours'].append(res['err_aci_ours'])
        agg_results['err_baseline'].append(res['err_baseline'])
        agg_results['err_dtaci'].append(res['err_dtaci'])
        
        agg_results['size_aci_ours'].append(res['size_aci_ours'])
        agg_results['size_baseline'].append(res['size_baseline'])
        agg_results['size_dtaci'].append(res['size_dtaci'])

        agg_results['alpha_aci_ours'].append(res['alpha_aci_ours'])
        agg_results['alpha_baseline'].append(res['alpha_baseline'])
        agg_results['alpha_dtaci'].append(res['alpha_dtaci'])
        
        agg_results['gamma_t'].append(res['gamma_t_history'])
        agg_results['gamma_dtaci'].append(res['gamma_dtaci_history'])
        
        agg_results['p_rmd'].append(res['p_rmd_history'])
        agg_results['p_gap'].append(res['p_gap_history'])
        
        for g, errs in res['err_aci_fixed'].items():
            agg_results['err_aci_fixed'][g].append(errs)
        for g, alphas in res['alpha_aci_fixed'].items():
            agg_results['alpha_aci_fixed'][g].append(alphas)
        for g, sizes in res['size_aci_fixed'].items():
            agg_results['size_aci_fixed'][g].append(sizes)
    
    print("Experiments Done.")
    
    # Simple save to verify it works, no heavy plotting here as that's in run_best_mmlu.py
    with open('jump_rmd_results.pkl', 'wb') as f:
         pickle.dump(agg_results, f)
