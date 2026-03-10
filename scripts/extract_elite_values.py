import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def get_exact_values(path, target_steps, metrics):
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    results = {step: {} for step in target_steps}
    
    for metric in metrics:
        if metric in ea.Tags()['scalars']:
            scalars = ea.Scalars(metric)
            steps = np.array([e.step for e in scalars])
            vals = np.array([e.value for e in scalars])
            for target in target_steps:
                # Interpolación para obtener el valor en el step exacto
                results[target][metric] = np.interp(target, steps, vals)
        else:
            for target in target_steps:
                results[target][metric] = np.nan
    return results

path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"

marked_steps = [645056, 645248, 645440, 645632, 645824, 646016, 646144, 646336, 646528, 646720, 646912, 647104, 647296, 647488, 647680, 647872, 648064, 648256, 648448, 648640, 648832, 649024, 649216, 649408, 649600, 649792, 649984, 650176, 650368, 650560, 650752, 650944, 651136, 651328, 651520]

metrics = [
    "custom/CPM", "custom/CPS", "custom/avg_center_dev", "custom/avg_speed", 
    "custom/collision_interval", "custom/collision_num", "custom/collision_rate", 
    "custom/collision_speed", "custom/episode_length", "custom/mean_reward", 
    "custom/routes_completed", "custom/total_distance", "custom/mean_steer_smoothness_x100", 
    "custom/total_reward", "replay_buffer/mean_recent_rewards", "replay_buffer/sum_recent_rewards", 
    "replay_buffer/mean_recent_steer_smoothness_x100", "rollout/ep_len_mean", "rollout/ep_gt_rew_mean"
]

print("Extrayendo valores de alta resolución...")
data = get_exact_values(path_ours, marked_steps, metrics)

# Formatear salida en tabla
for step in marked_steps:
    print(f"\n🚀 VALORES PARA STEP: {step:,}")
    print("-" * 60)
    for m in metrics:
        val = data[step].get(m, np.nan)
        print(f"  {m:<45} | {val:.6f}")

