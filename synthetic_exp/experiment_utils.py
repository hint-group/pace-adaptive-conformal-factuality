
import numpy as np
import scipy.stats as st
import pandas as pd
import sys
import os

try:
    from DtACI import DtACI
except ImportError:
    try:
        from .DtACI import DtACI
    except ImportError:
         print("Warning: Could not import DtACI in experiment_utils.py")

# --- Helper Functions ---
def get_model_quantile(q):
    """
    Returns the quantile of standard normal distribution.
    """
    return st.norm.ppf(q)

def get_true_alpha_star(mu_t, target_alpha):
    """
    Calculates the true optimal alpha given shift mu_t.
    """
    return 1.0 - st.norm.cdf(st.norm.ppf(1.0 - target_alpha) + mu_t)

def get_true_y(mu_t):
    """
    Generates a sample y from N(mu_t, 1).
    """
    return np.random.normal(loc=mu_t, scale=1.0)

def calc_recovery_lag(err_seq, jumps, target, tol=0.05, stable_k=50):
    """
    Calculates the recovery lag (number of steps to stabilize) after each jump.
    """
    w = 100
    rolling_err = pd.Series(err_seq).rolling(window=w, min_periods=1).mean().values
    
    lags = []
    lower = target - tol
    upper = target + tol
    
    for jump in jumps:
        t_start = jump['start']
        t_end = jump['end']
        limit = t_end - stable_k
        
        if limit <= t_start:
            lags.append(t_end - t_start)
            continue
            
        recovered_at = None
        # Check window starting at t
        for t in range(t_start, limit):
             # Ensure we stay in bounds
             chunk = rolling_err[t : t+stable_k]
             if np.all((chunk >= lower) & (chunk <= upper)):
                 recovered_at = t
                 break
        
        if recovered_at is not None:
            lags.append(recovered_at - t_start)
        else:
            lags.append(t_end - t_start)
            
    return np.mean(lags)

def run_aci_experiment_core(
    T,
    target_alpha,
    mu_trajectory,
    alpha_star_trajectory,
    y_full_local,
    gamma_fixed_list,
    window_size,
    state_a_mu,
    mu_threshold,
    dtaci_gammas,
    weight_param
):
    """
    Core logic for running a single ACI experiment.
    
    Args:
        T: Total time steps
        target_alpha: Target error rate
        mu_trajectory: Array of mean values for the data generation process
        alpha_star_trajectory: Array of optimal alpha values
        y_full_local: Array of generated observation data y
        gamma_fixed_list: List of gamma values for baseline ACI
        window_size: Window size for our method's error averaging
        state_a_mu: Mean value of the base state (in-distribution)
        mu_threshold: Threshold to determine in-distribution samples for calibration
        dtaci_gammas: List of scan steps for DtACI
        weight_param: Weight parameter for our method
    
    Returns:
        Dictionary containing trajectories of alpha, error, and internal variables.
    """
    
    # OOD Prob Calculation
    # 1. Compute mu_bg, sigma_bg using sliding window (past 20 samples)
    bg_window_size = 20
    series_y = pd.Series(y_full_local)
    mu_bg_series = series_y.shift(1).rolling(window=bg_window_size, min_periods=1).mean()
    std_bg_series = series_y.shift(1).rolling(window=bg_window_size, min_periods=1).std(ddof=1)
    
    # Fill missing
    is_base = np.abs(mu_trajectory - state_a_mu) < mu_threshold
    z_in = y_full_local[is_base]
    if len(z_in) == 0:
        in_domain_size = min(500, T // 4)
        z_in = y_full_local[:in_domain_size]
    
    mu_in_global = z_in.mean()
    sigma_in_global = max(z_in.std(ddof=1), 1e-6)
    
    mu_bg_filled = mu_bg_series.fillna(mu_in_global).values
    sigma_bg_filled = std_bg_series.fillna(sigma_in_global).values
    sigma_bg_filled = np.maximum(sigma_bg_filled, 1e-6)
    
    md_in = ((y_full_local - mu_in_global) / sigma_in_global)**2
    md_bg = ((y_full_local - mu_bg_filled) / sigma_bg_filled)**2
    rmd_raw = md_in - md_bg
    
    # 2. Dynamic Sigmoid Calibration
    
    if len(y_full_local[is_base]) == 0:
        # Fallback to first few
         in_domain_size = min(500, T // 4)
         # Create a mask for fallback
         is_base_fallback = np.zeros(T, dtype=bool)
         is_base_fallback[:in_domain_size] = True
         cal_rmds = rmd_raw[is_base_fallback]
    else:
         cal_rmds = rmd_raw[is_base]
    
    if len(cal_rmds) > 0:
        SIGMOID_CENTER = np.quantile(cal_rmds, 0.8)
    else:
        SIGMOID_CENTER = 3.0 
        
    SIGMOID_SLOPE = 10.0 / SIGMOID_CENTER if (SIGMOID_CENTER != 0 and not np.isnan(SIGMOID_CENTER)) else 1.0
    
    ood_prob_local = 1 / (1 + np.exp(-SIGMOID_SLOPE * (rmd_raw - SIGMOID_CENTER)))
    
    # Init
    alpha_baseline = np.full(T, target_alpha)
    alpha_aci_fixed = {g: np.zeros(T) for g in gamma_fixed_list}
    for g in gamma_fixed_list:
        alpha_aci_fixed[g][0] = target_alpha
    alpha_aci_ours = np.zeros(T)
    alpha_aci_ours[0] = target_alpha
    
    model_dtaci = DtACI(gammas=dtaci_gammas, alpha=target_alpha, eta=2.0, sigma=0.001)
    alpha_dtaci = np.zeros(T)
    alpha_dtaci[0] = model_dtaci.predict()
    
    p_rmd = ood_prob_local[0]
    
    err_aci_ours_list = []
    err_aci_fixed_list = {g: [] for g in gamma_fixed_list}
    err_baseline_list = []
    err_dtaci_list = []
    
    phi_t_history = []
    p_rmd_history = []
    p_gap_history = []
    gamma_t_history = []
    gamma_dtaci_history = []
    
    for t in range(T):
        alpha_star_t = alpha_star_trajectory[t]
        y_t = y_full_local[t]
        
        # Baseline
        q_baseline = get_model_quantile(1 - alpha_baseline[t])
        is_error_baseline = 1.0 if y_t > q_baseline else 0.0
        err_baseline_list.append(is_error_baseline)
        
        # Fixed
        for g in gamma_fixed_list:
            alpha_aci_fixed[g][t] = np.clip(alpha_aci_fixed[g][t], 0.0, 1.0)
            q_aci_fixed = get_model_quantile(1 - alpha_aci_fixed[g][t])
            is_error_fixed = 1.0 if y_t > q_aci_fixed else 0.0
            err_aci_fixed_list[g].append(is_error_fixed)

            if t < T-1:
                alpha_aci_fixed[g][t+1] = alpha_aci_fixed[g][t] + g * (target_alpha - is_error_fixed)
                alpha_aci_fixed[g][t+1] = np.clip(alpha_aci_fixed[g][t+1], 0.0, 1.0)
        
        # Ours
        alpha_aci_ours[t] = np.clip(alpha_aci_ours[t], 0.0, 1.0)
        q_aci_ours = get_model_quantile(1 - alpha_aci_ours[t])
        is_error_aci_ours = 1.0 if y_t > q_aci_ours else 0.0
        err_aci_ours_list.append(is_error_aci_ours)
        
        phi_t = (alpha_aci_ours[t] - alpha_star_t)**2
        phi_t_history.append(phi_t)
        
        p_rmd = 0.9 * p_rmd + 0.1 * ood_prob_local[t] # simple smoothing
        
        if t < window_size:
            avg_err = target_alpha
        else:
            avg_err = np.mean(err_aci_ours_list[t-window_size+1 : t+1])
        gap_t = abs(avg_err - target_alpha)
        GAP_SCALE = 1
        p_gap = min(1.0, gap_t / GAP_SCALE)

        weight = weight_param
        if t > 0:
            # We need history of p_rmd for the derivative term
            prev_p_rmd = p_rmd_history[t-1] if t-1 < len(p_rmd_history) else p_rmd 
            if len(p_rmd_history) > 0:
                 prev_p_rmd = p_rmd_history[-1] 
            else:
                 prev_p_rmd = p_rmd
            
            p_t = weight * abs(p_rmd - prev_p_rmd) + (1-weight) * p_gap
        else :
            p_t = weight * p_rmd + (1-weight) * p_gap
        
        gamma_t = p_t
        
        p_rmd_history.append(p_rmd)
        p_gap_history.append(p_gap)
        gamma_t_history.append(gamma_t)
        
        if t < T-1:
            alpha_aci_ours[t+1] = alpha_aci_ours[t] + gamma_t * (target_alpha - is_error_aci_ours)
            alpha_aci_ours[t+1] = np.clip(alpha_aci_ours[t+1], 0.0, 1.0)
        
        # DtACI
        if t > 0:
            alpha_dtaci[t] = model_dtaci.predict()
            
        q_dtaci = get_model_quantile(1 - alpha_dtaci[t])
        is_error_dtaci = 1.0 if y_t > q_dtaci else 0.0
        err_dtaci_list.append(is_error_dtaci)
        
        beta_t = np.clip(1.0 - st.norm.cdf(y_t), 1e-6, 1-1e-6)
        model_dtaci.update(beta_t)

        gamma_dtaci = model_dtaci.get_weighted_gamma()
        gamma_dtaci_history.append(gamma_dtaci)
    
    return {
        'alpha_aci_ours': alpha_aci_ours.copy(),
        'alpha_aci_fixed': {g: alpha_aci_fixed[g].copy() for g in gamma_fixed_list},
        'alpha_baseline': alpha_baseline.copy(),
        'alpha_dtaci': alpha_dtaci.copy(),
        'err_aci_ours': np.array(err_aci_ours_list),
        'err_aci_fixed': {g: np.array(err_aci_fixed_list[g]) for g in gamma_fixed_list},
        'err_baseline': np.array(err_baseline_list),
        'err_dtaci': np.array(err_dtaci_list),
        'phi_t_history': np.array(phi_t_history),
        'p_rmd_history': np.array(p_rmd_history),
        'p_gap_history': np.array(p_gap_history),
        'gamma_t_history': np.array(gamma_t_history),
        'gamma_dtaci_history': np.array(gamma_dtaci_history),
    }
