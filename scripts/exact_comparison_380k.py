import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def get_exact_values(path, target_step):
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    results = {}
    
    tags = ea.Tags()['scalars']
    for tag in tags:
        scalars = ea.Scalars(tag)
        steps = np.array([e.step for e in scalars])
        vals = np.array([e.value for e in scalars])
        # Interpolamos para obtener el valor exacto en el paso 380,000
        results[tag] = float(np.interp(target_step, steps, vals))
    return results

path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"
path_base = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0"

target_step = 380000

print(f"--- EXTRACCIÓN PASO {target_step:,} ---")
data_ours = get_exact_values(path_ours, target_step)
data_base = get_exact_values(path_base, target_step)

metrics_to_show = [
    ("Métricas de Desempeño (custom/)", [
        "custom/total_reward", "custom/routes_completed", "custom/total_distance", 
        "custom/avg_center_dev", "custom/avg_speed", "custom/mean_reward", 
        "custom/collision_rate", "custom/collision_num", "custom/episode_length", 
        "custom/mean_steer_smoothness_x100", "custom/CPS", "custom/CPM", 
        "custom/collision_interval", "custom/collision_speed"
    ]),
    ("Estadísticas Memoria (replay_buffer/)", [
        "replay_buffer/mean_recent_rewards", "replay_buffer/sum_recent_rewards", 
        "replay_buffer/mean_recent_steer_smoothness_x100"
    ]),
    ("Métricas Rollout (rollout/)", [
        "rollout/ep_gt_rew_mean", "rollout/ep_len_mean"
    ]),
    ("Entrenamiento SAC (train/)", [
        "train/actor_loss", "train/critic_loss", "train/ent_coef", 
        "train/ent_coef_loss", "train/learning_rate", "train/std"
    ])
]

for section, metrics in metrics_to_show:
    print(f"\n[{section}]")
    print(f"{'Métrica':<45} | {'OURS':<15} | {'BASELINE'}")
    print("-" * 80)
    for m in metrics:
        v_ours = data_ours.get(m, None)
        v_base = data_base.get(m, None)
        
        # Ocultar suavidad para Baseline según instrucción
        if "smoothness" in m:
            v_base_str = "---"
        else:
            v_base_str = f"{v_base:.6f}" if v_base is not None else "N/A"
            
        v_ours_str = f"{v_ours:.6f}" if v_ours is not None else "N/A"
        print(f"{m:<45} | {v_ours_str:<15} | {v_base_str}")

