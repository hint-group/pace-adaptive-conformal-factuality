
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
from tqdm import tqdm
import sys
import os

# Add local directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from experiment_utils import (
        run_aci_experiment_core, 
        get_true_alpha_star, 
        calc_recovery_lag,
        get_model_quantile
    )
    from DtACI import DtACI
except ImportError:
    # If running from root
    sys.path.append(os.path.join(os.getcwd(), 'synthetic_exp/v6-final'))
    from experiment_utils import (
        run_aci_experiment_core, 
        get_true_alpha_star, 
        calc_recovery_lag,
        get_model_quantile
    )
    from DtACI import DtACI

# --- Experiment Settings ---
T = 6000                 
TARGET_ALPHA = 0.1       

GAMMA_FIXED_LIST = [0.01, 0.2] # Baselines

# --- Our ACI-CF Settings ---
WINDOW_SIZE = 25        
STATE_A_MU = 0.0         
STATE_A_ID = 0           

JUMPS = [
    {'start': 1000, 'end': 1500, 'mu': 0.5,  'id': 1, 'label': 'Small Pos', 'color': 'lightblue'},
    {'start': 1500, 'end': 2000, 'mu': -0.5, 'id': 2, 'label': 'Small Neg', 'color': 'lightcoral'},
    {'start': 2000, 'end': 2500, 'mu': 1.0,  'id': 3, 'label': 'Small Pos', 'color': 'lightblue'},
    {'start': 2500, 'end': 3000, 'mu': -2.0, 'id': 4, 'label': 'Small Neg', 'color': 'lightcoral'},
    {'start': 3000, 'end': 3500, 'mu': 1.0,  'id': 5, 'label': 'Large Pos', 'color': 'lightgreen'},
    {'start': 3500, 'end': 4000, 'mu': -2.0, 'id': 6, 'label': 'Large Neg', 'color': 'lightyellow'},
    {'start': 4000, 'end': 4500, 'mu': 0.5,  'id': 7, 'label': 'Large Pos', 'color': 'lightgreen'},
    {'start': 4500, 'end': 5000, 'mu': -0.5, 'id': 8, 'label': 'Large Neg', 'color': 'lightyellow'},
]

rng = np.random.default_rng(42)

# --- Generate Trajectories ---
mu_trajectory = np.zeros(T)
alpha_star_trajectory = np.zeros(T)
target_alpha_array = np.full(T, TARGET_ALPHA)

mu_trajectory[:] = STATE_A_MU

for jump in JUMPS:
    mu_trajectory[jump['start']:jump['end']] = jump['mu']

for t in range(T):
    alpha_star_trajectory[t] = get_true_alpha_star(mu_trajectory[t], TARGET_ALPHA)

# --- Experiment Function ---
def run_single_experiment(seed=None, weight_param=0.5):
    if seed is not None:
        local_rng = np.random.default_rng(seed)
    else:
        local_rng = np.random.default_rng()
    
    y_full_local = np.zeros(T)
    for t in range(T):
        y_full_local[t] = local_rng.normal(loc=mu_trajectory[t], scale=1.0)
    
    gammas_dtaci = [0.001, 0.002, 0.004, 0.008, 0.0160, 0.032, 0.064, 0.128]
    
    return run_aci_experiment_core(
        T=T,
        target_alpha=TARGET_ALPHA,
        mu_trajectory=mu_trajectory,
        alpha_star_trajectory=alpha_star_trajectory,
        y_full_local=y_full_local,
        gamma_fixed_list=GAMMA_FIXED_LIST,
        window_size=WINDOW_SIZE,
        state_a_mu=STATE_A_MU,
        mu_threshold=0.1,
        dtaci_gammas=gammas_dtaci,
        weight_param=weight_param
    )

# --- Sensitivity Analysis ---
if __name__ == "__main__":
    N_EXPERIMENTS = 30
    WEIGHT_LIST = np.arange(0.0, 1.1, 0.1)
    
    # Store metrics: (Metric Name -> List of values corresponding to WEIGHT_LIST)
    results_ours = {
        'rmse_local_err': [],
        'recovery_time': [],
        'coverage': []
    }
    
    # Store baselines (Metric Name -> Dict {Method -> Value})
    results_baselines = {
        'rmse_local_err': {},
        'recovery_time': {},
        'coverage': {}
    }
    
    # Temporary storage for baseline accumulation (Method -> List of trials)
    baseline_acc_err_sq = {'Baseline': [], 'DtACI': []}
    baseline_acc_rec = {'Baseline': [], 'DtACI': []}
    baseline_acc_cov = {'Baseline': [], 'DtACI': []}
    for g in GAMMA_FIXED_LIST:
        baseline_acc_err_sq[f'ACI (\u03B3={g})'] = []
        baseline_acc_rec[f'ACI (\u03B3={g})'] = []
        baseline_acc_cov[f'ACI (\u03B3={g})'] = []
    
    print("Starting Sensitivity Analysis...")
    
    for w in WEIGHT_LIST:
        print(f"Weight: {w:.1f}")
        
        trials_local_err_sq_ours = []
        trials_rec_ours = []
        trials_cov_ours = []
        
        LOCAL_WINDOW_SIZE = 100
        
        for i in tqdm(range(N_EXPERIMENTS), leave=False):
            res = run_single_experiment(seed=42+i, weight_param=w)
            
            # --- Ours ---
            # 1. Local RMSE
            err_seq = res['err_aci_ours']
            rolling_err = pd.Series(err_seq).rolling(window=LOCAL_WINDOW_SIZE, min_periods=1).mean().values
            mse_local = np.mean((rolling_err - TARGET_ALPHA)**2)
            trials_local_err_sq_ours.append(mse_local)
            
            # 2. Recovery Time
            rec_time = calc_recovery_lag(err_seq, JUMPS, TARGET_ALPHA)
            trials_rec_ours.append(rec_time)
            
            # 3. Coverage
            cov = 1.0 - np.mean(err_seq)
            trials_cov_ours.append(cov)
            
            # --- Baselines (Only compute at w=0) ---
            if w == 0.0:
                methods = [
                    ('Baseline', res['err_baseline']),
                    ('DtACI', res['err_dtaci'])
                ]
                for g in GAMMA_FIXED_LIST:
                    methods.append((f'ACI (\u03B3={g})', res['err_aci_fixed'][g]))
                
                for name, err_arr in methods:
                    # RMSE
                    re = pd.Series(err_arr).rolling(window=LOCAL_WINDOW_SIZE, min_periods=1).mean().values
                    mse = np.mean((re - TARGET_ALPHA)**2)
                    baseline_acc_err_sq[name].append(mse)
                    
                    # Rec
                    rt = calc_recovery_lag(err_arr, JUMPS, TARGET_ALPHA)
                    baseline_acc_rec[name].append(rt)
                    
                    # Cov
                    c = 1.0 - np.mean(err_arr)
                    baseline_acc_cov[name].append(c)

        # Average Ours
        results_ours['rmse_local_err'].append(np.sqrt(np.mean(trials_local_err_sq_ours)))
        results_ours['recovery_time'].append(np.mean(trials_rec_ours))
        results_ours['coverage'].append(np.mean(trials_cov_ours))
        
        # Average Baselines (Once)
        if w == 0.0:
            for name in baseline_acc_err_sq.keys():
                results_baselines['rmse_local_err'][name] = np.sqrt(np.mean(baseline_acc_err_sq[name]))
                results_baselines['recovery_time'][name] = np.mean(baseline_acc_rec[name])
                results_baselines['coverage'][name] = np.mean(baseline_acc_cov[name])
    
    print("Analysis Complete.")