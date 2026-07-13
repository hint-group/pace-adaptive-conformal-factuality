
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import sem

# Add local directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import shared components
try:
    from experiment_utils import (
        run_aci_experiment_core, 
        get_true_alpha_star, 
        get_model_quantile,
        calc_recovery_lag
    )
    # Import settings directly from sensitivity script to ensure consistency
    from run_sensitivity_jump import (
        T, TARGET_ALPHA, GAMMA_FIXED_LIST, alpha_star_trajectory, JUMPS
    )
except ImportError:
    # If running from root
    sys.path.append(os.path.join(os.getcwd(), 'synthetic_exp/v6-final'))
    from experiment_utils import (
        run_aci_experiment_core, 
        get_true_alpha_star, 
        get_model_quantile,
        calc_recovery_lag
    )
    from run_sensitivity_jump import (
        T, TARGET_ALPHA, GAMMA_FIXED_LIST, alpha_star_trajectory, JUMPS
    )

from plot_utils_jump import plot_p_values, plot_alpha_trajectory, plot_local_error, plot_gamma_choice

# --- Config ---
BEST_WEIGHT = 0.3
N_EXPERIMENTS = 30
LOCAL_WINDOW_SIZE = 100

print(f"Running Best Weight Analysis with w={BEST_WEIGHT} for {N_EXPERIMENTS} experiments...")

# --- Experiment Function Wrapper ---

def run_single_experiment(seed=None, weight_param=0.5):
    if seed is not None:
        local_rng = np.random.default_rng(seed)
    else:
        local_rng = np.random.default_rng()
    
    from run_sensitivity_jump import mu_trajectory, STATE_A_MU, WINDOW_SIZE
    
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


# --- Storage ---
agg_results = {
    'phi_t_history': [], 
    'alpha_aci_ours': [],
    'alpha_aci_fixed': {g: [] for g in GAMMA_FIXED_LIST},
    'alpha_dtaci': [],
    
    'err_aci_ours': [],
    'err_aci_fixed': {g: [] for g in GAMMA_FIXED_LIST},
    'err_baseline': [],
    'err_dtaci': [],
    
    'p_rmd_history': [],
    'p_gap_history': [],
    'gamma_t_history': [],
    'gamma_dtaci_history': []
}

# --- Run Loop ---
for i in tqdm(range(N_EXPERIMENTS)):
    res = run_single_experiment(seed=42+i, weight_param=BEST_WEIGHT)
    
    agg_results['phi_t_history'].append(res['phi_t_history'])
    agg_results['alpha_aci_ours'].append(res['alpha_aci_ours'])
    agg_results['alpha_dtaci'].append(res['alpha_dtaci'])
    
    agg_results['err_aci_ours'].append(res['err_aci_ours'])
    agg_results['err_dtaci'].append(res['err_dtaci'])
    
    # Baseline
    agg_results['err_baseline'].append(res['err_baseline'])
    
    if 'gamma_t_history' in res: agg_results['gamma_t_history'].append(res['gamma_t_history'])
    if 'gamma_dtaci_history' in res: agg_results['gamma_dtaci_history'].append(res['gamma_dtaci_history'])
    if 'p_rmd_history' in res: agg_results['p_rmd_history'].append(res['p_rmd_history'])
    if 'p_gap_history' in res: agg_results['p_gap_history'].append(res['p_gap_history'])
    
    for g in GAMMA_FIXED_LIST:
        agg_results['alpha_aci_fixed'][g].append(res['alpha_aci_fixed'][g])
        agg_results['err_aci_fixed'][g].append(res['err_aci_fixed'][g])

# --- Helper Stats ---
def get_stats(data_list):
    arr = np.array(data_list)
    if len(arr) == 0: return np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    mean = np.mean(arr, axis=0)
    se = sem(arr, axis=0)
    ci = 1.96 * se
    return mean, ci, ci, np.std(arr, axis=0) # return mean, lower_ci_width, upper_ci_width, std

# --- Process Data for Plots ---

# 1. p values
mean_p_rmd, lo_p_rmd, hi_p_rmd, std_p_rmd = get_stats(agg_results['p_rmd_history'])
mean_p_gap, lo_p_gap, hi_p_gap, std_p_gap = get_stats(agg_results['p_gap_history'])

# 2. Alpha Trajectory
mean_aci_ours, lo_aci_ours, hi_aci_ours, std_aci_ours = get_stats(agg_results['alpha_aci_ours'])
mean_dtaci, lo_dtaci, hi_dtaci, std_dtaci = get_stats(agg_results['alpha_dtaci'])

mean_aci_fixed = {}
lo_aci_fixed = {}
hi_aci_fixed = {}
for g in GAMMA_FIXED_LIST:
    m, l, h, s = get_stats(agg_results['alpha_aci_fixed'][g])
    mean_aci_fixed[g] = m
    lo_aci_fixed[g] = m - l 
    hi_aci_fixed[g] = m + l

# 3. Local Error
def get_rolling_err(err_list, w=100):
    arr = np.array(err_list)
    mean_err_inst = np.mean(arr, axis=0)
    rolling_err = pd.Series(mean_err_inst).rolling(window=w, min_periods=1).mean().values
    
    rolling_trials = []
    for row in arr:
        r = pd.Series(row).rolling(window=w, min_periods=1).mean().values
        rolling_trials.append(r)
    rolling_trials = np.array(rolling_trials)
    std_rolling = np.std(rolling_trials, axis=0)
    return rolling_err, std_rolling

mean_local_err_aci_ours, std_local_err_aci_ours = get_rolling_err(agg_results['err_aci_ours'])
mean_local_err_dtaci, std_local_err_dtaci = get_rolling_err(agg_results['err_dtaci'])
mean_local_err_baseline, _ = get_rolling_err(agg_results['err_baseline']) 

mean_local_err_aci_fixed = {}
std_local_err_aci_fixed = {}
for g in GAMMA_FIXED_LIST:
    m, s = get_rolling_err(agg_results['err_aci_fixed'][g])
    mean_local_err_aci_fixed[g] = m
    std_local_err_aci_fixed[g] = s

# 4. Gamma Choice
mean_gamma_t, _, _, std_gamma_t = get_stats(agg_results['gamma_t_history'])
mean_gamma_dtaci, _, _, std_gamma_dtaci = get_stats(agg_results['gamma_dtaci_history'])


# --- Metrics Calculation (RMSE, NSE) ---
indices_shift = []
indices_full = list(range(T))
for jump in JUMPS:
    indices_shift.extend(range(jump['start'], jump['end']))
indices_shift = sorted(list(set(indices_shift)))

def calc_single_traj_metrics(traj, target, indices):
    if len(indices) == 0: return 0.0, 0.0
    traj_sub = traj[indices]
    if np.isscalar(target):
        target_sub = np.full(len(indices), target)
    else:
        target_sub = target[indices]
        
    mse = np.mean((traj_sub - target_sub)**2)
    rmse = np.sqrt(mse)
    
    numerator = np.sum((target_sub - traj_sub)**2)
    denominator = np.sum((target_sub - np.mean(target_sub))**2)
    nse = 1 - numerator/denominator if denominator != 0 else 0.0
    
    return rmse, nse

def calc_metrics_stats(trajectories_list, target, indices):
    rmses = []
    nses = []
    
    for traj in trajectories_list:
        r, n = calc_single_traj_metrics(np.array(traj), target, indices)
        rmses.append(r)
        nses.append(n)
        
    return (np.mean(rmses), np.std(rmses)), (np.mean(nses), np.std(nses))

rmse_full_ours_alpha, nse_full_ours_alpha = calc_metrics_stats(agg_results['alpha_aci_ours'], alpha_star_trajectory, indices_full)
rmse_shift_ours_alpha, nse_shift_ours_alpha = calc_metrics_stats(agg_results['alpha_aci_ours'], alpha_star_trajectory, indices_shift)

rmse_full_dtaci_alpha, nse_full_dtaci_alpha = calc_metrics_stats(agg_results['alpha_dtaci'], alpha_star_trajectory, indices_full)
rmse_shift_dtaci_alpha, nse_shift_dtaci_alpha = calc_metrics_stats(agg_results['alpha_dtaci'], alpha_star_trajectory, indices_shift)

rmse_full_fixed_alpha_dict = {}
nse_full_fixed_alpha_dict = {}
rmse_shift_fixed_alpha_dict = {}
nse_shift_fixed_alpha_dict = {}

for g in GAMMA_FIXED_LIST:
    r_f, n_f = calc_metrics_stats(agg_results['alpha_aci_fixed'][g], alpha_star_trajectory, indices_full)
    r_s, n_s = calc_metrics_stats(agg_results['alpha_aci_fixed'][g], alpha_star_trajectory, indices_shift)
    
    rmse_full_fixed_alpha_dict[g] = r_f 
    nse_full_fixed_alpha_dict[g] = n_f
    rmse_shift_fixed_alpha_dict[g] = r_s
    nse_shift_fixed_alpha_dict[g] = n_s

# Metrics For Local Error (Target = TARGET_ALPHA)
def calc_single_err_metrics(err_traj, target, indices):
    if len(indices) == 0: return 0.0, 0.0
    traj_sub = err_traj[indices]
    mse = np.mean((traj_sub - target)**2)
    rmse = np.sqrt(mse)
    violation = np.max(traj_sub - target)
    return rmse, violation

def calc_err_metrics_stats(err_trajectories_list, target, indices):
    rmses = []
    violations = []
    w = LOCAL_WINDOW_SIZE
    for raw_err_traj in err_trajectories_list:
        smooth_traj = pd.Series(raw_err_traj).rolling(window=w, min_periods=1).mean().values
        r, v = calc_single_err_metrics(smooth_traj, target, indices)
        rmses.append(r)
        violations.append(v)     
    return (np.mean(rmses), np.std(rmses)), (np.mean(violations), np.std(violations))

rmse_full_ours_err, worst_violation_ours = calc_err_metrics_stats(agg_results['err_aci_ours'], TARGET_ALPHA, indices_full)
rmse_shift_ours_err, _ = calc_err_metrics_stats(agg_results['err_aci_ours'], TARGET_ALPHA, indices_shift)

rmse_full_dtaci_err, worst_violation_dtaci = calc_err_metrics_stats(agg_results['err_dtaci'], TARGET_ALPHA, indices_full)
rmse_shift_dtaci_err, _ = calc_err_metrics_stats(agg_results['err_dtaci'], TARGET_ALPHA, indices_shift)

rmse_full_baseline_err, _ = calc_err_metrics_stats(agg_results['err_baseline'], TARGET_ALPHA, indices_full)
rmse_shift_baseline_err, _ = calc_err_metrics_stats(agg_results['err_baseline'], TARGET_ALPHA, indices_shift)

rmse_full_fixed_err_dict = {}
rmse_shift_fixed_err_dict = {}
worst_violation_fixed = {}

for g in GAMMA_FIXED_LIST:
    r_f, wv = calc_err_metrics_stats(agg_results['err_aci_fixed'][g], TARGET_ALPHA, indices_full)
    r_s, _ = calc_err_metrics_stats(agg_results['err_aci_fixed'][g], TARGET_ALPHA, indices_shift)
    
    rmse_full_fixed_err_dict[g] = r_f
    rmse_shift_fixed_err_dict[g] = r_s
    worst_violation_fixed[g] = wv


# --- Recovery Time / Adaptation Lag Analysis ---
print("\n" + "="*60)
print(f"RECOVERY TIME ANALYSIS (Mean +/- Std over {N_EXPERIMENTS} runs)")
print("="*60)

RECOVERY_TOLERANCE = 0.05
RECOVERY_K = 50
lower_bound = TARGET_ALPHA - RECOVERY_TOLERANCE
upper_bound = TARGET_ALPHA + RECOVERY_TOLERANCE

raw_err_map = {
    "Ours": agg_results['err_aci_ours'],
    "DtACI": agg_results['err_dtaci'],
    "Baseline": agg_results['err_baseline']
}
for g in GAMMA_FIXED_LIST:
    raw_err_map[f"Fixed({g})"] = agg_results['err_aci_fixed'][g]

print(f"Target Alpha: {TARGET_ALPHA}")
print(f"Acceptable Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Stability Condition: {RECOVERY_K} consecutive steps in range")

recovery_results = {name: {i: [] for i in range(len(JUMPS))} for name in raw_err_map.keys()}

for name, raw_err_list in raw_err_map.items():
    for raw_traj in raw_err_list:
        smooth_traj = pd.Series(raw_traj).rolling(window=LOCAL_WINDOW_SIZE, min_periods=1).mean().values
        
        for i, jump in enumerate(JUMPS):
            t_jump = jump['start']
            t_end = jump['end']
            limit = min(t_end, len(smooth_traj)) - RECOVERY_K
            
            if limit < t_jump:
                lag = t_end - t_jump 
                recovery_results[name][i].append(lag)
                continue
            
            recovered_at = None
            for t in range(t_jump, limit):
                window = smooth_traj[t : t + RECOVERY_K]
                if np.all((window >= lower_bound) & (window <= upper_bound)):
                    recovered_at = t
                    break
            
            if recovered_at is not None:
                lag = recovered_at - t_jump
            else:
                lag = t_end - t_jump
            
            recovery_results[name][i].append(lag)

# Print Report
for i, jump in enumerate(JUMPS):
    jump_label = jump.get('label', f'Jump {i+1}')
    print(f"\n{jump_label} (t={jump['start']}):")
    
    for name in raw_err_map.keys():
        lags = recovery_results[name][i]
        mu = np.mean(lags)
        sd = np.std(lags)
        print(f"  {name:<15}: {mu:.2f} +/- {sd:.2f} steps")

print("\n" + "="*60)
print("AVERAGE RECOVERY TIME ACROSS ALL JUMPS")
print("="*60)
for name in raw_err_map.keys():
    all_lags = []
    for i in range(len(JUMPS)):
        all_lags.extend(recovery_results[name][i])
    
    mu_global = np.mean(all_lags)
    sd_global = np.std(all_lags)
    print(f"{name:<15}: {mu_global:.2f} +/- {sd_global:.2f} steps")



# --- Metrics Printout ---
def print_metric_row(name, metric_tuple):
    print(f"{name:<15}: {metric_tuple[0]:.4f} +/- {metric_tuple[1]:.4f}")

print("\n" + "="*60)
print(f"METRICS ANALYSIS (Mean +/- Std over {N_EXPERIMENTS} experiments)")
print("="*60)

print("--- RMSE (Alpha vs Optimal) ---")
print("Full Trajectory:")
print_metric_row("Ours", rmse_full_ours_alpha)
print_metric_row("DtACI", rmse_full_dtaci_alpha)
for g in GAMMA_FIXED_LIST:
    print_metric_row(f"Fixed({g})", rmse_full_fixed_alpha_dict[g])

print("\nShift Segments Only:")
print_metric_row("Ours", rmse_shift_ours_alpha)
print_metric_row("DtACI", rmse_shift_dtaci_alpha)
for g in GAMMA_FIXED_LIST:
    print_metric_row(f"Fixed({g})", rmse_shift_fixed_alpha_dict[g])

print("\n--- RMSE (Error vs Target) ---")
print("Full Trajectory:")
print_metric_row("Ours", rmse_full_ours_err)
print_metric_row("DtACI", rmse_full_dtaci_err)
print_metric_row("Baseline", rmse_full_baseline_err)
for g in GAMMA_FIXED_LIST:
    print_metric_row(f"Fixed({g})", rmse_full_fixed_err_dict[g])

print("\nShift Segments Only:")
print_metric_row("Ours", rmse_shift_ours_err)
print_metric_row("DtACI", rmse_shift_dtaci_err)
for g in GAMMA_FIXED_LIST:
    print_metric_row(f"Fixed({g})", rmse_shift_fixed_err_dict[g])

print("\n--- Worst Violation (Max Error Deviation) ---")
print_metric_row("Ours", worst_violation_ours)
print_metric_row("DtACI", worst_violation_dtaci)
for g in GAMMA_FIXED_LIST:
    print_metric_row(f"Fixed({g})", worst_violation_fixed[g])

# Coverage 
def calc_avg_coverage_stats(err_trajectories_list, indices):
    covs = []
    for raw_err in err_trajectories_list:
        sub = raw_err[indices] if len(indices) > 0 else raw_err
        avg_err = np.mean(sub)
        covs.append(1.0 - avg_err)
    return np.mean(covs), np.std(covs)

print("\n--- Average Coverage (1 - Error Rate) ---")
cov_ours_full = calc_avg_coverage_stats(agg_results['err_aci_ours'], indices_full)
cov_dtaci_full = calc_avg_coverage_stats(agg_results['err_dtaci'], indices_full)
cov_baseline_full = calc_avg_coverage_stats(agg_results['err_baseline'], indices_full)
cov_fixed_full = {}
for g in GAMMA_FIXED_LIST:
    cov_fixed_full[g] = calc_avg_coverage_stats(agg_results['err_aci_fixed'][g], indices_full)

print(f"Target: {1 - TARGET_ALPHA:.2f}")
print_metric_row("Ours", cov_ours_full)
print_metric_row("DtACI", cov_dtaci_full)
print_metric_row("Baseline", cov_baseline_full)
for g in GAMMA_FIXED_LIST:
    print_metric_row(f"Fixed({g})", cov_fixed_full[g])