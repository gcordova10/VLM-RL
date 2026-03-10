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

# Umbrales que sugerimos antes
filters = [
    {"metric": "custom/CPM", "op": "<", "val": 10},
    {"metric": "custom/CPS", "op": "<", "val": 0.05},
    {"metric": "custom/avg_center_dev", "op": "<", "val": 0.15},
    {"metric": "custom/avg_speed", "op": ">", "val": 15},
    {"metric": "custom/collision_interval", "op": ">", "val": 1000},
    {"metric": "custom/collision_num", "op": "<", "val": 350},
    {"metric": "custom/collision_rate", "op": "<", "val": 0.95},
    {"metric": "custom/collision_speed", "op": "<", "val": 15},
    {"metric": "custom/episode_length", "op": ">", "val": 3000},
    {"metric": "custom/mean_reward", "op": ">", "val": 0.5},
    {"metric": "custom/routes_completed", "op": ">", "val": 9.0},
    {"metric": "custom/total_distance", "op": ">", "val": 2000},
    {"metric": "custom/total_reward", "op": ">", "val": 3000},
    {"metric": "replay_buffer/mean_recent_rewards", "op": ">", "val": 0.4},
    {"metric": "replay_buffer/sum_recent_rewards", "op": ">", "val": 150},
    {"metric": "rollout/ep_len_mean", "op": ">", "val": 2000},
    {"metric": "rollout/ep_gt_rew_mean", "op": ">", "val": 50}
]

print(f"Analizando {len(df)} pasos del Baseline...")

# Tomamos el mejor paso histórico (470630) y vemos por qué fallaría
step_check = 470630
if step_check in df.index:
    row = df.loc[step_check]
    print(f"\nAUDITORÍA DEL PASO {step_check}:")
    all_pass = True
    for f in filters:
        m = f['metric']
        if m in df.columns:
            val = row[m]
            passed = val < f['val'] if f['op'] == '<' else val > f['val']
            status = "✅ OK" if passed else "❌ FALLA"
            print(f"  {m}: {val:.4f} {f['op']} {f['val']} -> {status}")
            if not passed: all_pass = False
        else:
            print(f"  {m}: No encontrado en log")
    
    if all_pass: print("\n✨ EL PASO 470630 CUMPLE TODO.")
    else: print("\n❌ EL PASO 470630 NO CUMPLE.")

# Contador de fallos global
fail_counts = {f['metric']: 0 for f in filters}
for step, row in df.iterrows():
    for f in filters:
        m = f['metric']
        if m in df.columns:
            val = row[m]
            passed = val < f['val'] if f['op'] == '<' else val > f['val']
            if not passed: fail_counts[m] += 1

print("\n📊 ESTADÍSTICA DE 'MATANZA' (Cuántos pasos elimina cada filtro):")
for m, count in sorted(fail_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {m}: eliminó {count} de {len(df)} pasos")

