import pickle
import numpy as np

with open('sensitivity_results.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['results']
weights = results['weight']
facts = results['factuality']
rmses = results['rmse_recovery']
sizes = results['set_size']
times = results['recovery_time']

print("Sensitivity Analysis Results:")
print("Weight | Factuality | RMSE Recovery | Set Size | Recovery Time")
print("-------|------------|---------------|----------|--------------")

best_weight = -1
min_rmse = float('inf')

lines = []
lines.append("Weight,Factuality,RMSE,SetSize,RecoveryTime")

for i, w in enumerate(weights):
    print(f"{w:6.1f} | {facts[i]:10.2f} | {rmses[i]:13.4f} | {sizes[i]:8.2f} | {times[i]:13.1f}")
    lines.append(f"{w},{facts[i]},{rmses[i]},{sizes[i]},{times[i]}")
    
    if rmses[i] < min_rmse:
        min_rmse = rmses[i]
        best_weight = w

print(f"\nBest Weight (Min RMSE): {best_weight}")

with open('sensitivity_metrics_summary.csv', 'w') as f:
    f.write('\n'.join(lines))
