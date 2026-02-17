import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def get_scalars(path):
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags()['scalars']
    data = {}
    for tag in tags:
        scalars = ea.Scalars(tag)
        data[tag] = pd.DataFrame([{'step': event.step, 'value': event.value} for event in scalars])
    return data

path_baseline = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0"
path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"

base = get_scalars(path_baseline)
ours = get_scalars(path_ours)

metrics = [
    'custom/total_reward', 'custom/mean_reward', 'custom/routes_completed', 
    'custom/mean_steer_smoothness_x100', 'custom/total_distance', 'train/std', 'custom/CPM'
]

summary = []
for m in metrics:
    row = {'metric': m}
    if m in base:
        df = base[m]
        row['base_final'] = df['value'].iloc[-1]
        row['base_max'] = df['value'].max()
    else:
        row['base_final'] = row['base_max'] = np.nan

    if m in ours:
        df = ours[m]
        row['ours_final'] = df['value'].iloc[-1]
        row['ours_max'] = df['value'].max()
    else:
        row['ours_final'] = row['ours_max'] = np.nan
    summary.append(row)

df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))
