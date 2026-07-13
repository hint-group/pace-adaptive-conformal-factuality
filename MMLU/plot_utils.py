
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_rmd_ood(md_in, ood_prob, save_path=None):
    """
    Plots the Mahalanobis Distance to In-domain distribution and the resulting OOD Probability.
    """
    plt.figure(figsize=(15, 8))
    
    # MD to In-domain
    plt.subplot(2, 1, 1)
    plt.plot(md_in, label='MD to In-domain', color='blue', alpha=0.7)
    plt.title('Mahalanobis Distance to In-domain Distribution over Time')
    plt.xlabel('Time (t)')
    plt.ylabel('MD_in')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # OOD Probability
    plt.subplot(2, 1, 2)
    plt.plot(ood_prob, label='OOD Probability (from RMD)', color='green', alpha=0.7, linewidth=1.5)
    plt.title('OOD Probability (p_t) from RMD over Time')
    plt.xlabel('Time (t)')
    plt.ylabel('OOD Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved RMD plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_error_rates(error_data, window_size=100, target_alpha=0.1, save_path=None):
    """
    Plots instantaneous local error rates (rolling mean).
    
    Args:
        error_data (dict): Dictionary mapping method names to their binary error arrays (0 or 1).
                           e.g., {'Ours': [0, 1, 0...], 'Baseline': ...}
        window_size (int): Window size for rolling mean.
        target_alpha (float): Target error rate.
        save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(12, 6))
    
    colors = {
        'Ours': 'blue',
        'Baseline': 'red',
        'DtACI': 'purple'
    }
    
    for method_name, errors in error_data.items():
        # Compute rolling mean
        series = pd.Series(errors)
        rolling_err = series.rolling(window=window_size, min_periods=1).mean()
        
        color = colors.get(method_name)
        if color is None:
            # Handle Fixed Gamma keys or others
            if 'Fixed' in method_name:
                color = 'orange' if '0.01' in method_name else 'brown'
            else:
                color = 'gray'
                
        plt.plot(rolling_err, label=f'{method_name}', color=color, alpha=0.8, linewidth=1.5)

    plt.axhline(target_alpha, color='black', linestyle='--', linewidth=2, label=f'Target ({target_alpha})')
    
    plt.title(f'Instantaneous Local Error Rate (Rolling Window={window_size})')
    plt.xlabel('Time (t)')
    plt.ylabel('Local Error Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved Error Rate plot to {save_path}")
    else:
        plt.show()
    plt.close()

def get_recovery_time(err_series, jump_start, target_alpha, tolerance, k):
    """
    Calculates recovery time: how long until error stays within tolerance for k steps.
    """
    lower_bound = target_alpha - tolerance
    upper_bound = target_alpha + tolerance
    
    if jump_start >= len(err_series):
        return None
    
    # Check window starting from jump_start
    for t in range(jump_start, len(err_series) - k):
        window = err_series[t : t+k]
        if np.all((window >= lower_bound) & (window <= upper_bound)):
            return t - jump_start
            
    return None

def analyze_recovery_time(jumps, error_data, target_alpha=0.1, tolerance=0.01, k=50):
    """
    Prints recovery times for different methods at defined jump points.
    
    Args:
        jumps (dict): Dictionary {jump_name: start_time}.
        error_data (dict): Dictionary mapping method names to *rolling mean* error arrays.
    """
    print("\n--- Recovery Time Analysis ---")
    for jump_name, jump_time in jumps.items():
        print(f"Jump: {jump_name} (Time: {jump_time})")
        
        for method_name, err_series in error_data.items():
            rec_time = get_recovery_time(err_series, jump_time, target_alpha, tolerance, k)
            print(f"  {method_name}: {rec_time}")

def plot_sensitivity(weight_list, rmse_ours, rmse_baselines, save_path=None):
    """
    Plots sensitivity analysis results (RMSE vs Weight).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(weight_list, rmse_ours, marker='o', linestyle='-', linewidth=2, label='Our Method (Varying Weight)', color='blue')

    # Plot Baselines
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    baseline_items = list(rmse_baselines.items())
    
    for i, (name, val) in enumerate(baseline_items):
        color = colors[i % len(colors)]
        plt.axhline(val, linestyle='--', label=f'{name} ({val:.4f})', color=color)

    plt.title('Sensitivity Analysis: Local Error RMSE vs Weight')
    plt.xlabel('Weight (lambda)')
    plt.ylabel('RMSE (Local Error Tracking)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved Sensitivity plot to {save_path}")
    else:
        plt.show()
    plt.close()
