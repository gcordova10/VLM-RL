import pandas as pd
import numpy as np

def analyze_raw_telemetry(path, name):
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
        # Limpiamos filas que no son telemetría
        df = df[pd.to_numeric(df['step'], errors='coerce').notnull()].copy()
        df['steer'] = df['steer'].astype(float)
        df['throttle'] = df['throttle'].astype(float)
        df['speed'] = df['speed'].astype(float)
        df['center_dev'] = df['center_dev'].astype(float)
        
        # 1. Jitter Real (Suavidad del Volante)
        # Desviación estándar del cambio de steer entre pasos consecutivos
        steer_changes = df.groupby('episode')['steer'].diff().abs().dropna()
        steer_jerk = steer_changes.mean()
        
        # 2. Precisión de Navegación
        mean_dev = df['center_dev'].mean()
        
        # 3. Eficiencia Energética (Throttle Stability)
        throttle_jerk = df.groupby('episode')['throttle'].diff().abs().dropna().mean()
        
        # 4. Velocidad
        avg_speed = df['speed'].mean()
        
        # 5. Pasos recorridos
        total_steps = len(df)
        
        return {
            "name": name,
            "steer_jerk": steer_jerk,
            "mean_dev": mean_dev,
            "throttle_jerk": throttle_jerk,
            "avg_speed": avg_speed,
            "steps": total_steps
        }
    except Exception as e:
        return {"name": name, "error": str(e)}

path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/evaldense/model_300000_steps_eval.csv"
path_base = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/evaldense/model_300000_steps_eval.csv"

res_ours = analyze_raw_telemetry(path_ours, "OURS (CLG-Smooth)")
res_base = analyze_raw_telemetry(path_base, "BASELINE")

print(f"{'Métrica Física Real (Step 300k)':<35} | {'BASELINE':<15} | {'OURS':<15} | {'Mejora'}")
print("-" * 85)

metrics = [
    ("Steer Jerk (Oscilación Volante)", "steer_jerk", True),
    ("Precisión Centro Carril (m)", "mean_dev", True),
    ("Throttle Jerk (Tirones)", "throttle_jerk", True),
    ("Velocidad Media (km/h)", "avg_speed", False),
    ("Supervivencia (Steps totales)", "steps", False)
]

for label, key, lower_is_better in metrics:
    b = res_base[key]
    o = res_ours[key]
    diff = ((b - o) / b * 100) if lower_is_better else ((o - b) / b * 100)
    print(f"{label:<35} | {b:<15.6f} | {o:<15.6f} | {diff:>+7.2f}%")

