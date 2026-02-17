import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def get_full_df(path):
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags()['scalars']
    data_dict = {}
    for tag in tags:
        scalars = ea.Scalars(tag)
        # Agrupamos por step para evitar duplicados y tomamos la media
        df_tag = pd.DataFrame([{'step': e.step, 'val': e.value} for e in scalars])
        data_dict[tag] = df_tag.groupby('step')['val'].mean()
    
    return pd.DataFrame(data_dict).sort_index().interpolate(method='linear')

paths = {
    "Baseline": "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0",
    "Ours": "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"
}

for name, path in paths.items():
    print(f"\n{'='*20} ANALIZANDO: {name} {'='*20}")
    df = get_full_df(path)
    
    # Métricas de Éxito
    success_cols = ['custom/routes_completed', 'custom/total_reward', 'custom/total_distance']
    # Métricas de Calidad (Valles son mejores)
    quality_cols = ['custom/mean_steer_smoothness_x100', 'custom/avg_center_dev', 'custom/collision_rate']
    
    # 1. Encontrar el MOMENTO ELITE (Max Routes Completed)
    if 'custom/routes_completed' in df.columns:
        best_step = df['custom/routes_completed'].idxmax()
        elite_row = df.loc[[best_step]]
        print(f"\n🏆 MOMENTO ELITE (Paso {best_step}):")
        cols_to_show = [c for c in success_cols + quality_cols if c in df.columns]
        print(elite_row[cols_to_show].to_string())

    # 2. Intersección: ¿En qué paso tenemos Éxito > 0.8 Y Colisiones < 0.1?
    if 'custom/routes_completed' in df.columns and 'custom/collision_rate' in df.columns:
        intersection = df[(df['custom/routes_completed'] > 0.8) & (df['custom/collision_rate'] < 0.2)]
        if not intersection.empty:
            print(f"\n✨ INTERSECCIÓN ENCONTRADA (Éxito y Seguridad):")
            print(f"Total de pasos en intersección: {len(intersection)}")
            print("Muestra del mejor paso en la intersección:")
            print(intersection.sort_values(by='custom/routes_completed', ascending=False).head(1)[cols_to_show].to_string())
        else:
            print("\n❌ No hay intersección de alto éxito y baja colisión.")

