import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def get_values_at_steps(path, target_steps):
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags()['scalars']
    
    results = {step: {} for step in target_steps}
    
    for tag in tags:
        scalars = ea.Scalars(tag)
        steps = np.array([e.step for e in scalars])
        values = np.array([e.value for e in scalars])
        
        for target in target_steps:
            # Interpolación para obtener el valor exacto en el paso solicitado
            val = np.interp(target, steps, values)
            results[target][tag] = val
            
    return results

path_baseline = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0"
path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"

target_steps = [470630, 962229]

print("Extrayendo telemetría de alta resolución...")
base_data = get_values_at_steps(path_baseline, target_steps)
ours_data = get_values_at_steps(path_ours, target_steps)

all_tags = sorted(set(list(base_data[470630].keys()) + list(ours_data[470630].keys())))

# Crear tabla comparativa
rows = []
for tag in all_tags:
    rows.append({
        'Variable': tag,
        'Base_470k': base_data[470630].get(tag, np.nan),
        'Ours_470k': ours_data[470630].get(tag, np.nan),
        'Base_962k': base_data[962229].get(tag, np.nan),
        'Ours_962k': ours_data[962229].get(tag, np.nan)
    })

df = pd.DataFrame(rows)
print("\n--- COMPARATIVA TOTAL POR PASO DE ENTRENAMIENTO ---")
# Filtramos las más relevantes para la respuesta pero procesamos todas
relevant = [
    'custom/routes_completed', 'custom/total_reward', 'custom/total_distance',
    'custom/avg_center_dev', 'custom/mean_steer_smoothness_x100', 'custom/CPM',
    'train/std', 'custom/avg_speed'
]
print(df[df['Variable'].isin(relevant)].to_string(index=False))

# Guardar todo a CSV por si quieres el detalle completo
df.to_csv('scripts/full_comparison_steps.csv', index=False)
