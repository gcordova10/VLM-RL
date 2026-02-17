import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def get_mean_reward_series(path):
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    if 'custom/mean_reward' not in ea.Tags()['scalars']:
        return None
    scalars = ea.Scalars('custom/mean_reward')
    df = pd.DataFrame([{'step': e.step, 'value': e.value} for e in scalars])
    return df

path_baseline = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0"
path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"

df_base = get_mean_reward_series(path_baseline)
df_ours = get_mean_reward_series(path_ours)

def analyze_tranche(df, name):
    print(f"\n--- Análisis Detallado: {name} ---")
    # Valor Inicial
    print(f"Valor inicial: {df['value'].iloc[0]:.4f}")
    
    # Cruce por cero
    zero_cross = df[df['value'] >= 0].head(1)
    if not zero_cross.empty:
        print(f"Cruce por 0 (o positivo) en el paso: {zero_cross['step'].values[0]}")
    
    # Llegada a 0.6
    point_06 = df[df['value'] >= 0.6].head(1)
    if not point_06.empty:
        print(f"Primer alcance de 0.6 en el paso: {point_06['step'].values[0]}")
    else:
        print("Nunca alcanzó 0.6 de forma sostenida")

    # Alrededor de 800k
    tranche_800k = df[(df['step'] >= 750000) & (df['step'] <= 850000)]
    print(f"Mínimo cerca de 800k: {tranche_800k['value'].min():.4f}")
    print(f"Promedio cerca de 800k: {tranche_800k['value'].mean():.4f}")

    # Final
    print(f"Valor final (último step): {df['value'].iloc[-1]:.4f}")
    print(f"Promedio últimos 50k steps: {df[df['step'] > 950000]['value'].mean():.4f}")

analyze_tranche(df_base, "BASELINE")
analyze_tranche(df_ours, "OURS (CLG-SMOOTH)")

