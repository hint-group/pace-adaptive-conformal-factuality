
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
from tqdm import tqdm
import sys
import os
import pickle
import matplotlib.font_manager as fm

# Add local directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from experiment_utils import (
        run_aci_experiment_core, 
        get_true_alpha_star, 
        get_model_quantile
    )
    from DtACI import DtACI
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'synthetic_exp/'))
    from experiment_utils import (
        run_aci_experiment_core, 
        get_true_alpha_star, 
        get_model_quantile
    )
    from DtACI import DtACI


# --- Experiment Settings (from 2-Smooth-sensitivity.ipynb) ---
T = 6000                 
TARGET_ALPHA = 0.1       

GAMMA_FIXED_LIST = [0.01, 0.1] # Baselines

# --- Our ACI-CF Settings ---
GAMMA_MIN = 1e-9
GAMMA_MAX = 1
C_S = 1                
WINDOW_SIZE = 25 

# --- Simulation Settings: Base State ---
STATE_A_MU = 0.0         
STATE_A_ID = 0           

rng = np.random.default_rng(0) # Use same seed as notebook for trajectory generation

# --- Generate Trajectories (Smooth Shift Logic) ---
# Initialize mu_trajectory
mu_trajectory = np.zeros(T)
mu_trajectory[0] = STATE_A_MU  # μ₀ = 0

# Generate noise {ε_t} ~ N(0, 0.003)
EPSILON_VAR = 0.003
epsilon = rng.normal(loc=0.0, scale=np.sqrt(EPSILON_VAR), size=T)

# Recursive generation
curr_mu = 0.0
prev_jump = 0.0

for t in range(1, T):
    if t > 0:
        mu_trajectory[t] = curr_mu
    
    if t % 5 == 0:
        err = rng.normal(0, np.sqrt(EPSILON_VAR))
        new_mu = curr_mu + 0.5 * prev_jump + 0.5 * err
        prev_jump = new_mu - curr_mu
        curr_mu = new_mu
        mu_trajectory[t] = curr_mu

# Generate alpha_star trajectory
alpha_star_trajectory = np.zeros(T)
for t in range(T):
    alpha_star_trajectory[t] = get_true_alpha_star(mu_trajectory[t], TARGET_ALPHA)
alpha_star_trajectory = np.clip(alpha_star_trajectory, 0.0, 1.0)

target_alpha_array = np.full(T, TARGET_ALPHA)

# --- Experiment Function ---
def run_single_experiment(seed=None, weight_param=0.5):
    if seed is not None:
        local_rng = np.random.default_rng(seed)
    else:
        local_rng = np.random.default_rng()
    
    y_full_local = np.zeros(T)
    for t in range(T):
        y_full_local[t] = local_rng.normal(loc=mu_trajectory[t], scale=1.0)
    
    # In smooth shift, gamma list for DtACI is slightly different in original code
    gammas_dtaci = [0.001, 0.002, 0.004, 0.008, 0.0160, 0.032, 0.064, 0.128]
    
    # Note: mu_threshold was 0.2 in smooth shift vs 0.1 in jump.
    return run_aci_experiment_core(
        T=T,
        target_alpha=TARGET_ALPHA,
        mu_trajectory=mu_trajectory,
        alpha_star_trajectory=alpha_star_trajectory,
        y_full_local=y_full_local,
        gamma_fixed_list=GAMMA_FIXED_LIST,
        window_size=WINDOW_SIZE,
        state_a_mu=STATE_A_MU,
        mu_threshold=0.2, # Original value in run_sensitivity_smooth.py
        dtaci_gammas=gammas_dtaci,
        weight_param=weight_param
    )

# --- Sensitivity Analysis ---
if __name__ == "__main__":
    RESULTS_FILE = 'sensitivity_smooth_results.pkl'
    N_EXPERIMENTS = 30
    WEIGHT_LIST = np.arange(0.0, 1.1, 0.1)
    LOCAL_WINDOW_SIZE = 100 

    if os.path.exists(RESULTS_FILE):
        print(f"Loading results from {RESULTS_FILE}...")
        with open(RESULTS_FILE, 'rb') as f:
            saved_data = pickle.load(f)
            results_ours = saved_data['results_ours']
            results_baselines = saved_data['results_baselines']
    else:
        # Store metrics: (Metric Name -> List of values corresponding to WEIGHT_LIST)
        results_ours = {
            'rmse_local_err': [],
            'coverage': []
        }

        # Store baselines (Metric Name -> Dict {Method -> Value})
        results_baselines = {
            'rmse_local_err': {},
            'coverage': {}
        }

        # Temporary accumulation for baselines (Method -> List of trials)
        baseline_acc_err_sq = {'Baseline': [], 'DtACI': []}
        baseline_acc_cov = {'Baseline': [], 'DtACI': []}
        for g in GAMMA_FIXED_LIST:
            baseline_acc_err_sq[f'ACI (\u03B3={g})'] = []
            baseline_acc_cov[f'ACI (\u03B3={g})'] = []

        print("Starting Sensitivity Analysis (Smooth Shift)...")

        for w in WEIGHT_LIST:
            print(f"Weight: {w:.1f}")
            
            trials_local_err_sq_ours = []
            trials_cov_ours = []
            
            for i in tqdm(range(N_EXPERIMENTS), leave=False):
                res = run_single_experiment(seed=42+i, weight_param=w) 
                
                # --- Ours ---
                # 1. Local Error RMSE
                err_seq = res['err_aci_ours']
                rolling_err = np.convolve(err_seq, np.ones(LOCAL_WINDOW_SIZE)/LOCAL_WINDOW_SIZE, mode='valid')
                mse_local = np.mean((rolling_err - TARGET_ALPHA)**2)
                trials_local_err_sq_ours.append(mse_local)

                # 2. Coverage
                cov = 1.0 - np.mean(err_seq)
                trials_cov_ours.append(cov)
                
                # --- Baselines (Only compute at w=0) ---
                if w == 0.0:
                    # Prepare list of (name, err_array)
                    methods = [
                        ('Baseline', res['err_baseline']),
                        ('DtACI', res['err_dtaci'])
                    ]
                    for g in GAMMA_FIXED_LIST:
                        methods.append((f'ACI (\u03B3={g})', res['err_aci_fixed'][g]))
                    
                    for name, err_arr in methods:
                        # RMSE
                        re = np.convolve(err_arr, np.ones(LOCAL_WINDOW_SIZE)/LOCAL_WINDOW_SIZE, mode='valid')
                        mse = np.mean((re - TARGET_ALPHA)**2)
                        baseline_acc_err_sq[name].append(mse)

                        # Coverage
                        c = 1.0 - np.mean(err_arr)
                        baseline_acc_cov[name].append(c)

            # Average over trials for Ours
            results_ours['rmse_local_err'].append(np.sqrt(np.mean(trials_local_err_sq_ours)))
            results_ours['coverage'].append(np.mean(trials_cov_ours))
            
            # Average Baselines (Once)
            if w == 0.0:
                for name in baseline_acc_err_sq.keys():
                    results_baselines['rmse_local_err'][name] = np.sqrt(np.mean(baseline_acc_err_sq[name]))
                    results_baselines['coverage'][name] = np.mean(baseline_acc_cov[name])
        
        # Save results
        print(f"Saving results to {RESULTS_FILE}...")
        with open(RESULTS_FILE, 'wb') as f:
            pickle.dump({
                'results_ours': results_ours,
                'results_baselines': results_baselines
            }, f)

    print("Analysis Complete. Plotting...")

    # --- Plotting Function ---
    def plot_sensitivity(metric_key, y_label, title, filename, ax_limits=None):
        
        # Set Font to Times New Roman if available
        # You may want to update this absolute path or make it relative/configurable
        font_path = '../Times New Roman.ttf'
        try:
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'Times New Roman'
        except:
             pass
        
        if metric_key not in results_ours or not results_ours[metric_key]:
            print(f"Skipping {metric_key} (no data)")
            return

        fig, ax = plt.subplots(figsize=(8, 5.5))
        
        # Colors
        colors_map = {
            'Baseline': 'grey',       
            'DtACI': '#2ca02c',          
            'ACI (\u03B3=0.01)': '#ff7f0e', 
            'ACI (\u03B3=0.1)': '#d62728'   
        }
        
        # Plot Baselines
        for name, val in results_baselines[metric_key].items():
            if name == 'Baseline' and metric_key == 'rmse_local_err': continue

            color = colors_map.get(name, 'gray')
            label_txt = f'{name} ({val:.4f})'
            ax.axhline(y=val, linestyle='--', color=color, alpha=0.8, linewidth=3.5, 
                    label=label_txt)
        
        # Plot Ours
        y_vals = results_ours[metric_key]
        ax.plot(WEIGHT_LIST, y_vals, 
                marker='o', markersize=7, markeredgewidth=1.5, markeredgecolor='white',
                linestyle='-', linewidth=4.0, color='#1f77b4',
                label='PACE (Varying Weight)', zorder=10)

        # Style
        ax.set_xlabel(r'Weight', fontweight='bold', fontsize=25)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=25)
        
        ax.grid(True, linestyle=':', alpha=0.6, color='gray')
        
        loc = 'best'
        bbox = None
        if metric_key == 'rmse_local_err':
            bbox = (0.98, 0.85)
        elif metric_key == 'coverage':
            loc = 'lower right'
        
        ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=0.9, loc=loc, fontsize=15, ncol=2, bbox_to_anchor=bbox)
        
        ax.set_xlim(-0.05, 1.05)
        if ax_limits:
            ax.set_ylim(ax_limits)
            
        plt.tight_layout()
        plt.savefig(f'{filename}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{filename}.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")

    # 1. Local Error RMSE
    plot_sensitivity('rmse_local_err', 'RMSE (Local Error)', 'Sensitivity: Local Error', 'sensitivity_smooth_local_err')

    # 2. Coverage
    plot_sensitivity('coverage', 'Average Coverage', 'Sensitivity: Coverage', 'sensitivity_smooth_coverage')
    
    print("Done.")