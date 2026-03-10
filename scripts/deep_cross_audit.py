import pandas as pd
import numpy as np
import glob
import os
import re
from tensorboard.backend.event_processing import event_accumulator

def get_training_metrics(path, steps):
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags()['scalars']
    res = {int(s): {} for s in steps}
    for tag in ['custom/mean_reward', 'custom/mean_steer_smoothness_x100']:
        if tag in tags:
            scalars = ea.Scalars(tag)
            s_steps = np.array([e.step for e in scalars])
            s_vals = np.array([e.value for e in scalars])
            for s in steps:
                res[int(s)][tag] = float(np.interp(s, s_steps, s_vals))
    return res

def get_eval_metrics(eval_path, steps):
    res = {int(s): {} for s in steps}
    for s in steps:
        f = os.path.join(eval_path, f"model_{int(s)}_steps_eval_summary.csv")
        if os.path.exists(f):
            df = pd.read_csv(f)
            total = df[df['episode'] == 'total']
            if not total.empty:
                res[int(s)]['eval_success'] = float(total['success'].values[0])
                res[int(s)]['eval_routes'] = float(total['routes_completed'].values[0])
                res[int(s)]['eval_center_dev'] = float(total['center_dev_mean'].values[0])
    return res

# 100 checkpoints
steps = np.linspace(10000, 1000000, 100).astype(int)

# Rutas OURS
train_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"
eval_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/evaldense"

# Rutas BASELINE
train_base = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0"
eval_base = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/evaldense"

print("Auditoría OURS...")
tr_o = get_training_metrics(train_ours, steps)
ev_o = get_eval_metrics(eval_ours, steps)

print("Auditoría BASELINE...")
tr_b = get_training_metrics(train_base, steps)
ev_b = get_eval_metrics(eval_base, steps)

# Consolidar todo en un DataFrame gigante
rows = []
for s in steps:
    row = {"step": int(s)}
    # Ours
    row["Ours_Train_Rew"] = tr_o[s].get('custom/mean_reward', 0)
    row["Ours_Eval_Routes"] = ev_o[s].get('eval_routes', 0)
    row["Ours_Eval_Dev"] = ev_o[s].get('eval_center_dev', 1.0)
    # Base
    row["Base_Train_Rew"] = tr_b[s].get('custom/mean_reward', 0)
    row["Base_Eval_Routes"] = ev_b[s].get('eval_routes', 0)
    row["Base_Eval_Dev"] = ev_b[s].get('eval_center_dev', 1.0)
    rows.append(row)

df = pd.DataFrame(rows)
# Buscamos donde Ours es MEJOR que Base en Evaluacion Real (Mas rutas y menos desviacion)
winners = df[(df['Ours_Eval_Routes'] > df['Base_Eval_Routes']) & (df['Ours_Eval_Dev'] < df['Base_Eval_Dev'])]

print("\n--- PASOS DONDE NUESTRA MEJORA SUPERA AL BASELINE EN EVALUACIÓN REAL (DENSE) ---")
print(winners[['step', 'Ours_Eval_Routes', 'Base_Eval_Routes', 'Ours_Eval_Dev', 'Base_Eval_Dev']].head(10).to_string(index=False))

# Guardar a HTML para tu revisión
df.to_html('tensorboard_analysis/cross_audit_results.html')
