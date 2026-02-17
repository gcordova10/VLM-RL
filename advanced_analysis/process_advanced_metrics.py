import os
import pandas as pd
import glob
import numpy as np
import json
import re

def process_all_eval_folders():
    base_path = '/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl'
    output_dir = '/media/nemesis/disco4tb/Documents/VLM-RL/advanced_analysis/data'
    
    # Identificar carpetas de evaluación
    eval_folders = [d for d in os.listdir(base_path) if d.startswith('eval') and os.path.isdir(os.path.join(base_path, d))]
    
    all_data = []

    print(f"Encontradas {len(eval_folders)} carpetas de evaluación.")

    for folder in eval_folders:
        folder_path = os.path.join(base_path, folder)
        # Extraer etiquetas del nombre de la carpeta
        # ej: evaldenseTown05 -> Town: Town05, Traffic: dense
        town = "General"
        traffic = "Normal"
        if "Town01" in folder: town = "Town01"
        elif "Town03" in folder: town = "Town03"
        elif "Town04" in folder: town = "Town04"
        elif "Town05" in folder: town = "Town05"
        
        if "dense" in folder: traffic = "Dense"
        elif "empty" in folder: traffic = "Empty"

        csv_files = glob.glob(os.path.join(folder_path, 'model_*_steps_eval.csv'))
        print(f"  Procesando {folder} ({len(csv_files)} archivos)...")

        for f in csv_files:
            try:
                match = re.search(r'model_(\d+)_steps', os.path.basename(f))
                if not match: continue
                steps = int(match.group(1))

                # Leer CSV
                df = pd.read_csv(f, on_bad_lines='skip')
                df_clean = df[pd.to_numeric(df['steer'], errors='coerce').notnull()].copy()
                
                if df_clean.empty: continue
                
                df_clean['steer'] = df_clean['steer'].astype(float)
                df_clean['speed'] = df_clean['speed'].astype(float)

                # Calcular métricas
                steer_jitter = df_clean.groupby('episode')['steer'].std().mean()
                avg_speed = df_clean['speed'].mean()
                
                # Éxito (desde el summary correspondiente si es posible, o aproximado)
                # Por simplicidad aquí nos enfocamos en calidad
                
                all_data.append({
                    'steps': steps,
                    'town': town,
                    'traffic': traffic,
                    'folder': folder,
                    'jitter': round(float(steer_jitter), 4),
                    'speed': round(float(avg_speed), 2)
                })
            except:
                continue

    # Guardar resultados
    output_file = os.path.join(output_dir, 'advanced_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)
    
    print(f"Proceso finalizado. {len(all_data)} registros guardados en {output_file}")

if __name__ == "__main__":
    process_all_eval_folders()
