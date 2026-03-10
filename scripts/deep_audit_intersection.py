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
path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"

df_base = extract_tb(path_base)
df_ours = extract_tb(path_ours)

# Filtros definidos
filters = [
    {"m": "custom/CPM", "op": "<", "val": 8},
    {"m": "custom/CPS", "op": "<", "val": 0.005},
    {"m": "custom/avg_center_dev", "op": "<", "val": 0.1},
    {"m": "custom/avg_speed", "op": ">", "val": 15},
    {"m": "custom/collision_interval", "op": ">", "val": 5000},
    {"m": "custom/collision_num", "op": "<", "val": 230},
    {"m": "custom/collision_rate", "op": "<", "val": 0.8},
    {"m": "custom/collision_speed", "op": "<", "val": 4},
    {"m": "custom/episode_length", "op": ">", "val": 3000},
    {"m": "custom/mean_reward", "op": ">", "val": 0.5},
    {"m": "custom/routes_completed", "op": ">", "val": 4},
    {"m": "custom/total_distance", "op": ">", "val": 2000},
    {"m": "custom/mean_steer_smoothness_x100", "op": "<", "val": 15},
    {"m": "custom/total_reward", "op": ">", "val": 2000},
    {"m": "replay_buffer/mean_recent_rewards", "op": ">", "val": 0.5},
    {"m": "replay_buffer/sum_recent_rewards", "op": ">", "val": 200},
    {"m": "replay_buffer/mean_recent_steer_smoothness_x100", "op": "<", "val": 8},
    {"m": "rollout/ep_len_mean", "op": ">", "val": 3000},
    {"m": "rollout/ep_gt_rew_mean", "op": ">", "val": 100}
]

def analyze_model(df, name):
    print(f"\n--- AUDITORÍA: {name} ---")
    valid_steps_all = None
    
    for f in filters:
        metric = f['m']
        if metric not in df.columns:
            print(f"[{metric}] -> No disponible en este modelo. Ignorando filtro.")
            continue
            
        val = df[metric]
        condition = val < f['val'] if f['op'] == '<' else val > f['val']
        steps = df.index[condition].tolist()
        
        if steps:
            print(f"[{metric}] -> Cumple en {len(steps)} pasos. Rango: {min(steps)} a {max(steps)}")
            current_valid = set(steps)
            if valid_steps_all is None:
                valid_steps_all = current_valid
            else:
                valid_steps_all = valid_steps_all.intersection(current_valid)
        else:
            print(f"[{metric}] -> ❌ NUNCA CUMPLE EL UMBRAL.")
            valid_steps_all = set()

    print(f"\n🎯 INTERSECCIÓN FINAL ({name}):")
    if valid_steps_all:
        print(f"✅ Se han encontrado {len(valid_steps_all)} pasos que cumplen TODOS los criterios.")
        sorted_steps = sorted(list(valid_steps_all))
        print(f"Pasos destacados: {sorted_steps[:10]} ...")
    else:
        print("❌ Ningún paso del entrenamiento cumple todos los criterios simultáneamente.")

analyze_model(df_base, "BASELINE")
analyze_model(df_ours, "OURS (CLG-SMOOTH)")
