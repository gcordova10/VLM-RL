import os
import pandas as pd
import glob
import numpy as np
import json
import re
import sys

def process_all_eval_folders(base_path, output_file):
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
        elif "Town02" in folder: town = "Town02"
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
                
                # Check if 'steer' and 'speed' columns exist
                if 'steer' not in df.columns or 'speed' not in df.columns:
                    continue

                df_clean = df[pd.to_numeric(df['steer'], errors='coerce').notnull()].copy()
                
                if df_clean.empty: continue
                
                df_clean['steer'] = df_clean['steer'].astype(float)
                df_clean['speed'] = df_clean['speed'].astype(float)

                # Calcular métricas
                steer_jitter = df_clean.groupby('episode')['steer'].std().mean()
                avg_speed = df_clean['speed'].mean()
                
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

    # Asegurar que el directorio de salida existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Guardar resultados
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)
    
    print(f"Proceso finalizado. {len(all_data)} registros guardados en {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 advanced_analysis/process_advanced_metrics.py <BASE_PATH> <OUTPUT_JSON>")
        sys.exit(1)
        
    process_all_eval_folders(sys.argv[1], sys.argv[2])
