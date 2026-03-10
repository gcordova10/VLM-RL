import pandas as pd
import os

# Ruta al archivo de resumen de evaluación estándar (Town 2, Regular Density)
# Este escenario es el que el paper reporta en la Tabla 2
file_path = '/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/eval/model_1000000_steps_eval_summary.csv'

def extract_table_2_vlmrl():
    print(f"🔍 Leyendo datos para Tabla 2 desde: {file_path}")
    
    if not os.path.exists(file_path):
        print("❌ Error: No se encuentra el archivo de evaluación.")
        return

    df = pd.read_csv(file_path)
    
    # Obtenemos la fila 'total' que contiene los promedios de los 10 episodios
    row = df[df['episode'] == 'total']
    
    # Mapeo directo a las columnas de la Tabla 2 del Paper
    metrics = {
        "Modelo": "VLM-RL (Local)",
        "AS ↑ (Velocidad Media)": round(float(row['speed_mean'].values[0]), 2),
        "RC ↑ (Completitud Ruta)": round(float(row['routes_completed'].values[0]), 2),
        "TD ↑ (Distancia Total)": round(float(row['total_distance'].values[0]), 2),
        "CS ↓ (Vel. Colisión)": round(float(row['collision_speed'].values[0]), 2),
        "SR ↑ (Tasa de Éxito)": round(float(row['success'].values[0]), 2)
    }
    
    # Mostrar tabla
    df_table2 = pd.DataFrame([metrics])
    print("========================================================================")
    print("DATOS LOCALES PARA TABLE 2 (ESCENARIO: TOWN 02, TRÁFICO REGULAR)")
    print("========================================================================")
    print(df_table2.to_string(index=False))
    print("========================================================================")

if __name__ == "__main__":
    extract_table_2_vlmrl()
