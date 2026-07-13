
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm

# Ensure imports work from current dir
sys.path.append(os.getcwd())
try:
    from mmlu_utils import get_rolling, safe_rmse, get_recovery_time
    from mmlu import run_single_experiment_mmlu, T, TARGET_ALPHA, GAMMA_FIXED_LIST, STABLE_SEGMENT_SIZE, JUMP_SEGMENT_SIZE, NUM_CYCLES
except ImportError:
    # If running from inside MMLU_v8_score/MMLU
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from mmlu_utils import get_rolling, safe_rmse, get_recovery_time
    from mmlu import run_single_experiment_mmlu, T, TARGET_ALPHA, GAMMA_FIXED_LIST, STABLE_SEGMENT_SIZE, JUMP_SEGMENT_SIZE, NUM_CYCLES

# Set Font to Times New Roman
try:
    font_path = '../Times New Roman.ttf'
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Times New Roman'
except Exception as e:
    print(f"Warning: font loading failed: {e}")

COLORS = {
    'ours':      'blue',  
    'fixed_0.01': 'orange', 
    'dtaci':     'green',  
    'fixed_0.1': 'red',  
    'gt':        'black'   
}

# ==========================================
# Configuration
# ==========================================
BEST_WEIGHT = 0.4 
N_EXPERIMENTS = 30
LOCAL_WINDOW_SIZE = 100

def calc_metrics_for_method(error_list, size_list):
    # 1. Factuality (1 - Global Error)
    err_means_per_trial = np.mean(np.array(error_list), axis=1)
    fact_val = (1.0 - err_means_per_trial) * 100
    fact_stats = (np.mean(fact_val), np.std(fact_val))
    
    # 2. Coverage (1 - Error) vs Target
    cov_val = 1.0 - err_means_per_trial
    cov_stats = (np.mean(cov_val), np.std(cov_val))
    
    # 3. RMSE (Global, Shift, Stable)
    curr = 0
    shift_indices = []
    stable_indices = []
    jump_starts = []
    
    for i in range(NUM_CYCLES):
        # Stable
        end = min(curr + STABLE_SEGMENT_SIZE, T)
        if end > curr:
            stable_indices.extend(range(curr, end))
        curr = end
        # Jump
        end = min(curr + JUMP_SEGMENT_SIZE, T)
        if end > curr:
            shift_indices.extend(range(curr, end))
            jump_starts.append(curr)
        curr = end
        
    shift_indices = np.array(shift_indices)
    stable_indices = np.array(stable_indices)

    # 1. Loop for Std
    rmse_global_list = []
    rmse_shift_list = []
    rmse_stable_list = []
    
    for trial_err in error_list:
        smooth = get_rolling(trial_err, w=LOCAL_WINDOW_SIZE)
        
        # Global 
        rmse_global_list.append(np.sqrt(np.mean((smooth - TARGET_ALPHA)**2)))
        
        # Shift 
        if len(shift_indices) > 0:
            valid = shift_indices[shift_indices < len(smooth)]
            rmse_shift_list.append(safe_rmse(smooth, TARGET_ALPHA, valid))
        else:
            rmse_shift_list.append(0.0)
            
        # Stable
        if len(stable_indices) > 0:
            valid = stable_indices[stable_indices < len(smooth)]
            rmse_stable_list.append(safe_rmse(smooth, TARGET_ALPHA, valid))
        else:
            rmse_stable_list.append(0.0)

    # 2. Aggregated Mean for Mean RMSE
    def calc_rmse_of_aggregated_mean(error_list, indices=None, target=TARGET_ALPHA):
        mean_curve = np.mean(np.array(error_list), axis=0) 
        smooth_mean_curve = get_rolling(mean_curve, w=LOCAL_WINDOW_SIZE)
        
        if indices is not None and len(indices) > 0:
            valid_indices = indices[indices < len(smooth_mean_curve)]
            if len(valid_indices) == 0: return 0.0
            selected_data = smooth_mean_curve[valid_indices]
        else:
            selected_data = smooth_mean_curve
            
        rmse_val = np.sqrt(np.mean((selected_data - target)**2))
        return rmse_val

    # Global
    mean_val_global = calc_rmse_of_aggregated_mean(error_list, indices=None)
    std_val_global = np.std(rmse_global_list) 
    rmse_global_stats = (mean_val_global, std_val_global)
    
    # Shift
    mean_val_shift = calc_rmse_of_aggregated_mean(error_list, indices=shift_indices)
    std_val_shift = np.std(rmse_shift_list) 
    rmse_shift_stats = (mean_val_shift, std_val_shift)
    
    # Stable
    mean_val_stable = calc_rmse_of_aggregated_mean(error_list, indices=stable_indices)
    std_val_stable = np.std(rmse_stable_list)
    rmse_stable_stats = (mean_val_stable, std_val_stable)
    
    # 4. Recovery Time
    rec_time_means = []
    for trial_err in error_list:
        smooth = get_rolling(trial_err, w=LOCAL_WINDOW_SIZE)
        rt_list = []
        for j_start in jump_starts:
            rt = get_recovery_time(smooth, j_start, TARGET_ALPHA)
            if rt is not None:
                rt_list.append(rt)
        if rt_list:
            rec_time_means.append(np.mean(rt_list))
            
    if rec_time_means:
        rec_stats = (np.mean(rec_time_means), np.std(rec_time_means))
    else:
        rec_stats = (np.nan, np.nan)
        
    # 5. Set Size
    size_means = np.mean(np.array(size_list), axis=1)
    size_stats = (np.mean(size_means), np.std(size_means))
    
    return {
        'factuality': fact_stats,
        'coverage': cov_stats,
        'rmse_global': rmse_global_stats,
        'rmse_shift': rmse_shift_stats,
        'rmse_stable': rmse_stable_stats,
        'recovery_time': rec_stats,
        'set_size': size_stats
    }


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print(f"Running Best Weight Analysis with w={BEST_WEIGHT} for {N_EXPERIMENTS} experiments (MMLU)...")

    agg_results = {
        'err_aci_ours': [], 'size_aci_ours': [], 'gamma_t': [], 'p_rmd': [], 'p_gap': [],
        # Baselines
        'err_baseline': [], 'size_baseline': [],
        'err_dtaci': [], 'gamma_dtaci': [], 'size_dtaci': [],
        'err_aci_fixed': {g: [] for g in GAMMA_FIXED_LIST},
        'size_aci_fixed': {g: [] for g in GAMMA_FIXED_LIST}
    }

    for i in tqdm(range(N_EXPERIMENTS)):
        res = run_single_experiment_mmlu(seed=1000+i, weight=BEST_WEIGHT) # Different seed range
        
        agg_results['err_aci_ours'].append(res['err_aci_ours'])
        agg_results['size_aci_ours'].append(res['size_aci_ours'])
        agg_results['gamma_t'].append(res['gamma_t_history'])
        agg_results['p_rmd'].append(res['p_rmd_history'])
        agg_results['p_gap'].append(res['p_gap_history'])
        
        agg_results['err_baseline'].append(res['err_baseline'])
        agg_results['size_baseline'].append(res['size_baseline'])
        
        agg_results['err_dtaci'].append(res['err_dtaci'])
        agg_results['size_dtaci'].append(res['size_dtaci'])
        agg_results['gamma_dtaci'].append(res['gamma_dtaci_history'])
        
        for g, errs in res['err_aci_fixed'].items():
            agg_results['err_aci_fixed'][g].append(errs)
        for g, sizes in res['size_aci_fixed'].items():
            agg_results['size_aci_fixed'][g].append(sizes)

    # --- Metrics ---
    methods_to_analyze = {
        'Ours': (agg_results['err_aci_ours'], agg_results['size_aci_ours']),
        'Baseline': (agg_results['err_baseline'], agg_results['size_baseline']),
        'DtACI': (agg_results['err_dtaci'], agg_results['size_dtaci']),
    }
    for g in GAMMA_FIXED_LIST:
        methods_to_analyze[f'Fixed_{g}'] = (agg_results['err_aci_fixed'][g], agg_results['size_aci_fixed'][g])

    
    print("\n" + "="*120)
    print(f"{'Method':<15} | {'Factuality (%)':<20} | {'RMSE (Global)':<20} | {'RMSE (Shift)':<20} | {'Set Size':<15} | {'Rec Time':<15}")
    print("-" * 120)

    for name, (errs, sizes) in methods_to_analyze.items():
        m = calc_metrics_for_method(errs, sizes)
        
        fact_str = f"{m['factuality'][0]:.2f} +/- {m['factuality'][1]:.2f}"
        rmse_g_str = f"{m['rmse_global'][0]:.4f} +/- {m['rmse_global'][1]:.4f}"
        rmse_s_str = f"{m['rmse_shift'][0]:.4f} +/- {m['rmse_shift'][1]:.4f}"
        size_str = f"{m['set_size'][0]:.2f} +/- {m['set_size'][1]:.2f}"
        rec_str = f"{m['recovery_time'][0]:.1f} +/- {m['recovery_time'][1]:.1f}"
        
        print(f"{name:<15} | {fact_str:<20} | {rmse_g_str:<20} | {rmse_s_str:<20} | {size_str:<15} | {rec_str:<15}")
    
    print("="*120)

    # --- Plotting ---
    print("Generating plots...")
    
    def mean_axis0(key): return np.mean(agg_results[key], axis=0)
    
    mean_err_ours = mean_axis0('err_aci_ours')
    mean_err_base = mean_axis0('err_baseline')
    mean_err_dtaci = mean_axis0('err_dtaci')
    mean_gamma_t = mean_axis0('gamma_t')
    mean_gamma_dtaci = mean_axis0('gamma_dtaci')
    mean_p_rmd = mean_axis0('p_rmd')
    mean_p_gap = mean_axis0('p_gap')
    
    mean_err_fixed = {g: np.mean(agg_results['err_aci_fixed'][g], axis=0) for g in GAMMA_FIXED_LIST}
    
    # 1. Local Error (Styled with Std)
    def get_rolling_stats(key_or_list):
        if isinstance(key_or_list, str):
            raw_list = agg_results[key_or_list]
        else:
            raw_list = key_or_list
        rolled = np.array([get_rolling(arr, w=LOCAL_WINDOW_SIZE) for arr in raw_list])
        return np.mean(rolled, axis=0), np.std(rolled, axis=0)

    mean_ours, std_ours = get_rolling_stats('err_aci_ours')
    mean_dtaci, std_dtaci = get_rolling_stats('err_dtaci')

    plt.figure(figsize=(8, 3))

    plt.plot(mean_ours, label="PACE (Ours)", color=COLORS['ours'], linewidth=2, alpha=0.9)
    plt.fill_between(range(len(mean_ours)), mean_ours - std_ours, mean_ours + std_ours, color=COLORS['ours'], alpha=0.2)
    
    for g in GAMMA_FIXED_LIST:
        mean_fixed, _ = get_rolling_stats(agg_results['err_aci_fixed'][g])
        color = COLORS['fixed_0.01'] if g == 0.01 else COLORS['fixed_0.1']
        plt.plot(mean_fixed, label=f"ACI ($\\gamma={g}$)", color=color, linewidth=2, alpha=0.7, linestyle='--')

    plt.plot(mean_dtaci, label="DtACI", color=COLORS['dtaci'], linewidth=2, alpha=0.7, linestyle='-.')
    plt.fill_between(range(len(mean_dtaci)), mean_dtaci - std_dtaci, mean_dtaci + std_dtaci, color='darkorange', alpha=0.2)
    
    plt.axhline(TARGET_ALPHA, color='black', linestyle='--', linewidth=2, label=f'Target $\\alpha={TARGET_ALPHA}$', zorder=0)

    plt.xlabel("Time (t)", fontsize=14)
    plt.ylabel("Local Error Rate", fontsize=14)
    plt.legend(fontsize=9.5, framealpha=0.8, ncol=5, loc='upper center')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.ylim(0.025, 0.22)
    plt.tight_layout()
    plt.savefig('mmlu_best_local_error.png')
    plt.savefig('mmlu_best_local_error.pdf', bbox_inches='tight')

    # 2. Gamma / P values
    plt.figure(figsize=(15, 6))
    plt.plot(mean_p_rmd, label='p_rmd', color='green', linestyle='--')
    plt.plot(mean_p_gap, label='p_gap', color='orange', linestyle=':')
    plt.plot(mean_gamma_t, label='gamma_t (ours)', color='blue', linewidth=2)
    plt.title(f"MMLU: Gamma Components (w={BEST_WEIGHT})")
    plt.legend()
    plt.savefig('mmlu_best_gamma_components.png')
    
    print("Done.")
