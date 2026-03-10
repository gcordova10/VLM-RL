import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def extract_tb(path):
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    data = {}
    for tag in ea.Tags()['scalars']:
        scalars = ea.Scalars(tag)
        data[tag] = pd.DataFrame([{'step': e.step, 'val': e.value} for e in scalars]).groupby('step')['val'].mean()
    return pd.DataFrame(data)

path_base = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0"
df = extract_tb(path_base)

# Métricas que tenemos en el Baseline
metrics = [
    'custom/CPM', 'custom/CPS', 'custom/avg_center_dev', 'custom/avg_speed', 
    'custom/collision_interval', 'custom/collision_num', 'custom/collision_rate', 
    'custom/collision_speed', 'custom/episode_length', 'custom/mean_reward', 
    'custom/routes_completed', 'custom/total_distance', 'custom/total_reward',
    'replay_buffer/mean_recent_rewards', 'replay_buffer/sum_recent_rewards',
    'rollout/ep_len_mean', 'rollout/ep_gt_rew_mean'
]

print("--- VALORES PARA LIBERAR EL BASELINE (Mínimos exigibles para ver 1 paso) ---")
for m in metrics:
    if m in df.columns:
        if any(x in m for x in ['CPM', 'CPS', 'center_dev', 'collision_num', 'collision_rate', 'collision_speed']):
            # Para estas el umbral debe ser MAYOR que el mínimo
            print(f"{m}: Poner Threshold > {df[m].max():.4f} (actualmente es muy bajo)")
        else:
            # Para estas el umbral debe ser MENOR que el máximo
            print(f"{m}: Poner Threshold < {df[m].min():.4f} (actualmente es muy alto)")

# Buscamos el "Momento de Oro" del Baseline por routes_completed
best_step = df['custom/routes_completed'].idxmax()
elite_data = df.loc[best_step]

print(f"\n🏆 CONFIGURACIÓN PARA VER EL MEJOR PASO ({best_step}) DEL BASELINE:")
for m in metrics:
    if m in df.columns:
        val = elite_data[m]
        op = "<" if any(x in m for x in ['CPM', 'CPS', 'center_dev', 'collision_num', 'collision_rate', 'collision_speed']) else ">"
        # Ajustamos un margen del 5% para que se vea
        suggested = val * 1.05 if op == "<" else val * 0.95
        print(f"  * {m}: {op} {suggested:.4f}")

