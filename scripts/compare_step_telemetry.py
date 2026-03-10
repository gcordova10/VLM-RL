import pandas as pd
import numpy as np

def analyze_csv(path, name):
    df = pd.read_csv(path, on_bad_lines='skip')
    # Limpiar filas que no son telemetría (como las de 'route')
    df = df[pd.to_numeric(df['step'], errors='coerce').notnull()].copy()
    df['steer'] = df['steer'].astype(float)
    df['throttle'] = df['throttle'].astype(float)
    df['speed'] = df['speed'].astype(float)
    df['center_dev'] = df['center_dev'].astype(float)
    
    # Calcular métricas de calidad de control
    # Smoothness: Cambio medio entre pasos (Jerk de dirección)
    df['steer_diff'] = df.groupby('episode')['steer'].diff().abs()
    steer_smoothness = df['steer_diff'].mean()
    
    # Jitter: Desviación estándar del steer por episodio
    steer_jitter = df.groupby('episode')['steer'].std().mean()
    
    # Precisión de carril
    mean_dev = df['center_dev'].mean()
    max_dev = df['center_dev'].max()
    
    # Velocidad
    avg_speed = df['speed'].mean()
    
    return {
        "name": name,
        "steer_smoothness": steer_smoothness,
        "steer_jitter": steer_jitter,
        "mean_center_dev": mean_dev,
        "max_center_dev": max_dev,
        "avg_speed": avg_speed,
        "total_steps": len(df)
    }

path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/eval/model_380000_steps_eval.csv"
path_base = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/eval/model_380000_steps_eval.csv"

res_ours = analyze_csv(path_ours, "OURS (CLG-Smooth)")
res_base = analyze_csv(path_base, "BASELINE")

print(f"{'Métrica':<30} | {'BASELINE':<15} | {'OURS':<15} | {'Mejora'}")
print("-" * 80)
metrics = [
    ("Steer smoothness (Jerk)", "steer_smoothness", True),
    ("Steer Jitter (StdDev)", "steer_jitter", True),
    ("Mean Center Dev (m)", "mean_center_dev", True),
    ("Max Center Dev (m)", "max_center_dev", True),
    ("Avg Speed (km/h)", "avg_speed", False),
]

for label, key, lower_is_better in metrics:
    b = res_base[key]
    o = res_ours[key]
    diff = ((b - o) / b * 100) if lower_is_better else ((o - b) / b * 100)
    print(f"{label:<30} | {b:<15.6f} | {o:<15.6f} | {diff:>+7.2f}%")

