import pandas as pd
import os

def get_paper_vs_local_table2():
    # Valores extraídos del Paper (Tabla 2: VLM-RL Ours)
    # Formato: Valor ± Desviación
    paper_data = {
        "Métrica": ["AS ↑ (Velocidad)", "RC ↑ (Completitud)", "TD ↑ (Distancia)", "CS ↓ (Colisión)", "SR ↑ (Éxito)"],
        "VLM-RL (Paper)": ["19.3 ± 1.29", "0.97 ± 0.03", "2028.2 ± 96.6", "0.02 ± 0.03", "0.93 ± 0.04"]
    }

    # Ruta al archivo local
    local_csv = '/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/eval/model_1000000_steps_eval_summary.csv'
    
    local_values = []
    if os.path.exists(local_csv):
        df = pd.read_csv(local_csv)
        row = df[df['episode'] == 'total']
        if not row.empty:
            local_values = [
                f"{round(float(row['speed_mean'].values[0]), 2)}",
                f"{round(float(row['routes_completed'].values[0]), 2)}",
                f"{round(float(row['total_distance'].values[0]), 2)}",
                f"{round(float(row['collision_speed'].values[0]), 2)}",
                f"{round(float(row['success'].values[0]), 2)}"
            ]
        else:
            local_values = ["N/A"] * 5
    else:
        local_values = ["Archivo no encontrado"] * 5

    # Crear DataFrame comparativo
    comparison_df = pd.DataFrame({
        "Métrica": paper_data["Métrica"],
        "VLM-RL (Paper)": paper_data["VLM-RL (Paper)"],
        "VLM-RL (Tu Local)": local_values
    })

    print("\n" + "="*85)
    print("COMPARATIVA TABLA 2: VLM-RL (PAPER) VS EVALUACIÓN LOCAL")
    print("="*85)
    print(comparison_df.to_string(index=False))
    print("="*85)
    print("\nNOTA: Los valores del paper son promedios de múltiples pruebas.")
    print(f"Los valores locales provienen de: {local_csv}")

if __name__ == "__main__":
    get_paper_vs_local_table2()