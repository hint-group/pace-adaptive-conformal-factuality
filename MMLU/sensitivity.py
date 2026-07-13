
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys

# Style Setup
try:
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
    from plot_style import set_pub_style
    set_pub_style()
    print("Loaded publication plot style.")
except ImportError:
    print("Warning: plot_style not found. Using default style.")
    plt.style.use('seaborn-v0_8-muted' if 'seaborn-v0_8-muted' in plt.style.available else 'default')

# Import Local Modules
try:
    from mmlu_utils import get_rolling, safe_rmse, get_recovery_time
    from mmlu import run_single_experiment_mmlu, TARGET_ALPHA, STABLE_SEGMENT_SIZE, JUMP_SEGMENT_SIZE, NUM_CYCLES, T, GAMMA_FIXED_LIST
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from mmlu_utils import get_rolling, safe_rmse, get_recovery_time
    from mmlu import run_single_experiment_mmlu, TARGET_ALPHA, STABLE_SEGMENT_SIZE, JUMP_SEGMENT_SIZE, NUM_CYCLES, T, GAMMA_FIXED_LIST

# ==========================================
# Configuration
# ==========================================
WEIGHTS = np.round(np.arange(0.0, 1.1, 0.1), 1)
TRIALS_PER_WEIGHT = 30 
WINDOW_SIZE = 50
RESULTS_FILE = 'sensitivity_results.pkl'

def compute_metrics(agg_results):
    mean_err_ours = np.mean(np.array(agg_results['err_aci_ours']), axis=1) # Shape (T,)
    global_factuality = (1.0 - np.mean(mean_err_ours)) * 100

    mean_err_ours = np.mean(np.array(agg_results['err_aci_ours']), axis=0)
    mean_size_ours = np.mean(agg_results['size_aci_ours'])
    
    curr = 0
    shift_indices = []
    jump_starts = []
    for i in range(NUM_CYCLES):
        end = min(curr + STABLE_SEGMENT_SIZE, T)
        curr = end
        end = min(curr + JUMP_SEGMENT_SIZE, T)
        if end > curr:
            shift_indices.extend(range(curr, end))
            jump_starts.append(curr)
        curr = end
    shift_indices = np.array(shift_indices)
    
    mean_local_err_ours = get_rolling(mean_err_ours, w=WINDOW_SIZE)
    global_rmse = safe_rmse(mean_local_err_ours, TARGET_ALPHA, np.arange(len(mean_local_err_ours)))
    shift_rmse = safe_rmse(mean_local_err_ours, TARGET_ALPHA, shift_indices)
    
    rec_times = []
    for jump_time in jump_starts:
        rt = get_recovery_time(mean_local_err_ours, jump_time, TARGET_ALPHA)
        if rt is not None:
            rec_times.append(rt)
    
    avg_recovery_time = np.mean(rec_times) if rec_times else np.nan
    retained_rate = mean_size_ours / 4.0
    
    return global_factuality, global_rmse, shift_rmse, mean_size_ours, avg_recovery_time, retained_rate


# ==========================================
# Data Loading / Execution
# ==========================================
if os.path.exists(RESULTS_FILE):
    print(f"Loading cached results from {RESULTS_FILE}...")
    with open(RESULTS_FILE, 'rb') as f:
        data = pickle.load(f)
        results = data['results']
        baseline_metrics = data['baselines']
        
    if 'retained_rate' not in results:
        results['retained_rate'] = [s / 4.0 for s in results['set_size']]
    
    for k in list(baseline_metrics.keys()):
        val = baseline_metrics[k]
        if len(val) == 4:
            baseline_metrics[k] = val + (val[2]/4.0,)

else:
    print(f"No cache found. Starting Sensitivity Analysis on Weights: {WEIGHTS}")
    results = {
        'weight': [],
        'factuality': [],
        'global_rmse': [],
        'shift_rmse': [],
        'set_size': [],
        'recovery_time': [],
        'retained_rate': []
    }
    baseline_metrics = {}

    for w in WEIGHTS:
        print(f"\nProcessing Weight = {w}...")
        
        agg_w = {
            'err_aci_ours': [], 'size_aci_ours': [],
            'err_baseline': [], 'size_baseline': [], 'err_dtaci': [], 'size_dtaci': [],
            'err_aci_fixed': {g: [] for g in GAMMA_FIXED_LIST}, 'size_aci_fixed': {g: [] for g in GAMMA_FIXED_LIST}
        }
        
        for i in range(TRIALS_PER_WEIGHT):
            res = run_single_experiment_mmlu(seed=42+i, weight=w)
            agg_w['err_aci_ours'].append(res['err_aci_ours'])
            agg_w['size_aci_ours'].append(res['size_aci_ours'])
            
            if w == WEIGHTS[0]:
                agg_w['err_baseline'].append(res['err_baseline'])
                agg_w['size_baseline'].append(res['size_baseline'])
                agg_w['err_dtaci'].append(res['err_dtaci'])
                agg_w['size_dtaci'].append(res['size_dtaci'])
                for g, errs in res['err_aci_fixed'].items():
                    agg_w['err_aci_fixed'][g].append(errs)
                for g, sizes in res['size_aci_fixed'].items():
                    agg_w['size_aci_fixed'][g].append(sizes)

        fact, global_rmse, shift_rmse, size, rectime, ret_rate = compute_metrics(agg_w)
        print(f"  Result: Fact={fact:.2f}%, Global RMSE={global_rmse:.4f}, Shift RMSE={shift_rmse:.4f}, Size={size:.2f}, RecTime={rectime:.1f}")
        
        results['weight'].append(w)
        results['factuality'].append(fact)
        results['global_rmse'].append(global_rmse)
        results['shift_rmse'].append(shift_rmse)
        results['set_size'].append(size)
        results['recovery_time'].append(rectime)
        results['retained_rate'].append(ret_rate)
        
        if w == WEIGHTS[0]:
            print("  Computing Baseline Metrics...")
            def get_base_metrics(errs, sizes):
                mock = {'err_aci_ours': errs, 'size_aci_ours': sizes}
                return compute_metrics(mock)
                
            baseline_metrics['Baseline'] = get_base_metrics(agg_w['err_baseline'], agg_w['size_baseline'])
            baseline_metrics['DtACI'] = get_base_metrics(agg_w['err_dtaci'], agg_w['size_dtaci'])
            
            for g in GAMMA_FIXED_LIST:
                 baseline_metrics[f'Fixed_{g}'] = get_base_metrics(agg_w['err_aci_fixed'][g], agg_w['size_aci_fixed'][g])

    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump({'results': results, 'baselines': baseline_metrics}, f)


# ==========================================
# Plotting using FActScore Style
# ==========================================
print("\nPlotting...")

colors_map = {
    'Baseline': 'gray',          
    'DtACI': '#2ca02c',          
    'ACI ($\gamma=0.01$)': '#ff7f0e',     
    'ACI ($\gamma=0.1$)': '#d62728',     
}

metrics_config = [
    {
        'key': 'factuality',
        'title': 'Factuality (%)',
        'idx': 0,
        'ylabel': 'Factuality (%)',
        'ylim': None,
        'filename': 'sensitivity_factuality'
    },
    {
        'key': 'global_rmse',
        'title': 'RMSE',
        'idx': 1,
        'ylabel': 'RMSE (Local Error)',
        'ylim': None,
        'filename': 'sensitivity_rmse'
    },
    {
        'key': 'set_size',
        'title': 'Set Size',
        'idx': 2,
        'ylabel': 'Average Set Size',
        'ylim': None,
        'filename': 'sensitivity_set_size'
    },
    {
        'key': 'retained_rate',
        'title': 'Retained Rate',
        'idx': 4,
        'ylabel': 'Retained Rate',
        'ylim': None,
        'filename': 'sensitivity_retained_rate'
    }
]

for config in metrics_config:
    fig, ax = plt.subplots(figsize=(8, 5.5))  

    for base_name, base_vals in baseline_metrics.items():
        val = base_vals[config['idx']]
        
        label_name = base_name
        color_key = base_name
        
        if 'Fixed_' in base_name:
             gamma_val_str = base_name.split('_')[1]
             gamma_val = float(gamma_val_str)
             
             if abs(gamma_val - 0.1) < 1e-5:
                 label_name = 'ACI ($\gamma=0.1$)'
             elif abs(gamma_val - 0.01) < 1e-5:
                 label_name = 'ACI ($\gamma=0.01$)'
             else:
                 label_name = f'ACI ($\gamma$={gamma_val})'
             
             color_key = label_name
             
        elif base_name == 'Baseline':
            continue
        
        color = colors_map.get(color_key, 'black') 
        
        label_text = label_name
        
        ax.axhline(y=val, linestyle='--', color=color, alpha=0.8, linewidth=2, label=label_text)

    ax.plot(results['weight'], results[config['key']], 
            marker='o', markersize=9, markeredgewidth=1.5, markeredgecolor='white',
            linestyle='-', linewidth=3, color='#1f77b4', 
            label='Ours (Varying Weight)', zorder=10)

    ax.set_xlabel(r'Weight', fontweight='bold', fontsize=20)
    ax.set_ylabel(config['ylabel'], fontweight='bold', fontsize=20)
    
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax.set_xlim(-0.05, 1.05)
    
    if config['ylim']:
        ax.set_ylim(config['ylim'])

    ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=0.9, fontsize=12, loc='best')

    plt.tight_layout()
    
    png_path = f"{config['filename']}.png"
    pdf_path = f"{config['filename']}.pdf"
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved {png_path} and {pdf_path}")
    
    plt.close()
