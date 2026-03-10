import pandas as pd
import os

# Ruta base de tu experimento
base_path = '/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl'
# Checkpoint que queremos analizar (el final del entrenamiento)
checkpoint = 'model_1000000_steps_eval_summary.csv'

print(f"📂 Analizando experimento en: {base_path}")
print(f"🎯 Usando checkpoint: {checkpoint}\n")

# Mapeo de Tablas del Paper a carpetas generadas por run_eval.py
# Tabla 4 del Paper: Generalización en Ciudades
files_towns = {
    'Town 1': os.path.join(base_path, 'evalTown01', checkpoint),
    'Town 2': os.path.join(base_path, 'eval', checkpoint),        # Town 2 es la carpeta por defecto 'eval'
    'Town 3': os.path.join(base_path, 'evalTown03', checkpoint),
    'Town 4': os.path.join(base_path, 'evalTown04', checkpoint),
    'Town 5': os.path.join(base_path, 'evalTown05', checkpoint),
}

# Tabla 5 del Paper: Generalización en Densidades (en Town 2)
files_density = {
    'Empty':   os.path.join(base_path, 'evalempty', checkpoint),
    'Regular': os.path.join(base_path, 'eval', checkpoint),        # Regular es la carpeta por defecto 'eval'
    'Dense':   os.path.join(base_path, 'evaldense', checkpoint),
}

def get_data_from_csv(name, filepath):
    """Lee el CSV y extrae la última fila (resumen total)"""
    if not os.path.exists(filepath):
        return {'Escenario': name, 'AS': '-', 'RC': '-', 'TD': '-', 'CS': '-', 'SR': '-', 'Estado': 'No encontrado'}
    
    try:
        # Leemos el CSV generado por eval_plots.py -> summary_eval()
        df = pd.read_csv(filepath)
        
        # Buscamos la fila que tiene el promedio total (creada en eval_plots.py línea 160)
        # Normalmente tiene 'episode' = 'total' o es la última fila
        row = df[df['episode'] == 'total']
        if row.empty:
             row = df.iloc[[-1]]
        
        # Extraemos y redondeamos los valores
        return {
            'Escenario': name,
            'AS': round(float(row['speed_mean'].values[0]), 2),
            'RC': round(float(row['routes_completed'].values[0]), 2),
            'TD': round(float(row['total_distance'].values[0]), 2),
            'CS': round(float(row['collision_speed'].values[0]), 2),
            'SR': round(float(row['success'].values[0]), 2),
            'Estado': 'OK'
        }
    except Exception as e:
        return {'Escenario': name, 'AS': '-', 'RC': '-', 'TD': '-', 'CS': '-', 'SR': '-', 'Estado': f'Error: {str(e)}'}

# --- Generar Tabla 4 ---
print("="*80)
print("TABLA 4: GENERALIZACIÓN EN DIFERENTES CIUDADES (TOWNS)")
print("="*80)
data_towns = [get_data_from_csv(k, v) for k, v in files_towns.items()]
df_towns = pd.DataFrame(data_towns)
print(df_towns.to_string(index=False))
print("\n")

# --- Generar Tabla 5 ---
print("="*80)
print("TABLA 5: GENERALIZACIÓN EN DENSIDADES DE TRÁFICO (Town 2)")
print("="*80)
data_density = [get_data_from_csv(k, v) for k, v in files_density.items()]
df_density = pd.DataFrame(data_density)
print(df_density.to_string(index=False))