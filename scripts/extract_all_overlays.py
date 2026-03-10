import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import json

def get_raw_scalars(path):
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    data = {}
    for tag in ea.Tags()['scalars']:
        scalars = ea.Scalars(tag)
        data[tag] = [{'step': e.step, 'val': e.value} for e in scalars]
    return data

def get_ranges(steps):
    if not steps: return []
    steps = sorted(list(set(steps)))
    ranges = []
    start = steps[0]
    for i in range(1, len(steps)):
        if steps[i] > steps[i-1] + 5000: # Salto de 5k pasos define nuevo bloque
            ranges.append({"start": start, "end": steps[i-1]})
            start = steps[i]
    ranges.append({"start": start, "end": steps[-1]})
    return ranges

paths = {
    "baseline": "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0",
    "ours": "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"
}

# Umbrales configurados
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

full_report = {}

for name, path in paths.items():
    print(f"Procesando {name}...")
    raw_data = get_raw_scalars(path)
    model_report = {}
    
    for f in filters:
        metric = f['m']
        if metric in raw_data:
            condition_steps = []
            for d in raw_data[metric]:
                pass_cond = d['val'] < f['v'] if f['op'] == '<' else d['val'] > f['v']
                if pass_cond:
                    condition_steps.append(d['step'])
            
            model_report[metric] = {
                "count": len(condition_steps),
                "ranges": get_ranges(condition_steps)
            }
        else:
            model_report[metric] = {"count": 0, "ranges": []}
            
    full_report[name] = model_report

output_file = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard_analysis/overlay_episodes.json"
with open(output_file, "w") as f:
    json.dump(full_report, f, indent=4)

print(f"\n✅ Auditoría completa guardada en: {output_file}")
