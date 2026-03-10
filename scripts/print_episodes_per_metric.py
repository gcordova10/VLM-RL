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

def get_ranges(steps):
    if not steps: return "Ninguno"
    steps = sorted(steps)
    ranges = []
    if not steps: return ""
    start = steps[0]
    for i in range(1, len(steps)):
        if steps[i] > steps[i-1] + 1000: # Tolerancia de 1000 pasos (nuestra resolución de ffill)
            ranges.append(f"[{start:,} - {steps[i-1]:,}]")
            start = steps[i]
    ranges.append(f"[{start:,} - {steps[-1]:,}]")
    return ", ".join(ranges)

path_base = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0"
path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"

df_base = extract_tb(path_base).reindex(range(0, 1000001, 1000)).ffill().bfill()
df_ours = extract_tb(path_ours).reindex(range(0, 1000001, 1000)).ffill().bfill()

filters = [
    {"m": "custom/CPM", "op": "<", "v": 8},
    {"m": "custom/CPS", "op": "<", "v": 0.005},
    {"m": "custom/avg_center_dev", "op": "<", "v": 0.1},
    {"m": "custom/avg_speed", "op": ">", "v": 15},
    {"m": "custom/collision_interval", "op": ">", "v": 5000},
    {"m": "custom/collision_num", "op": "<", "v": 230},
    {"m": "custom/collision_rate", "op": "<", "v": 0.8},
    {"m": "custom/collision_speed", "op": "<", "v": 4},
    {"m": "custom/episode_length", "op": ">", "v": 3000},
    {"m": "custom/mean_reward", "op": ">", "v": 0.5},
    {"m": "custom/routes_completed", "op": ">", "v": 4},
    {"m": "custom/total_distance", "op": ">", "v": 2000},
    {"m": "custom/mean_steer_smoothness_x100", "op": "<", "v": 15},
    {"m": "custom/total_reward", "op": ">", "v": 2000},
    {"m": "replay_buffer/mean_recent_rewards", "op": ">", "v": 0.5},
    {"m": "replay_buffer/sum_recent_rewards", "op": ">", "v": 200},
    {"m": "replay_buffer/mean_recent_steer_smoothness_x100", "op": "<", "v": 8},
    {"m": "rollout/ep_len_mean", "op": ">", "v": 3000},
    {"m": "rollout/ep_gt_rew_mean", "op": ">", "v": 100}
]

print(f"{'MÉTRICA':<45} | {'OURS (PASOS)':<40} | {'BASELINE (PASOS)'}")
print("-" * 120)

for f in filters:
    metric = f['m']
    op = f['op']
    val = f['v']
    
    # Ours
    steps_ours = []
    if metric in df_ours.columns:
        cond = df_ours[metric] < val if op == '<' else df_ours[metric] > val
        steps_ours = df_ours.index[cond].tolist()
    
    # Baseline
    steps_base = []
    if metric in df_base.columns:
        cond = df_base[metric] < val if op == '<' else df_base[metric] > val
        steps_base = df_base.index[cond].tolist()
    
    print(f"{metric:<45} | {get_ranges(steps_ours):<40} | {get_ranges(steps_base)}")

