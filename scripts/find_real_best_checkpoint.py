import pandas as pd
import glob
import os
import re

base_path = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/eval"
files = glob.glob(os.path.join(base_path, "*_summary.csv"))

results = []
for f in files:
    match = re.search(r"model_(\d+)_steps", os.path.basename(f))
    if not match: continue
    step = int(match.group(1))
    
    df = pd.read_csv(f)
    total = df[df['episode'] == 'total']
    if total.empty: continue
    
    # Extraemos métricas físicas reales de este checkpoint
    success = float(total['success'].values[0])
    routes = float(total['routes_completed'].values[0])
    center_dev = float(total['center_dev_mean'].values[0])
    # Como smoothness no está en el summary, usamos el proxy de la tabla que viste
    # pero aquí priorizamos éxito y precisión.
    
    results.append({
        "step": step,
        "success": success,
        "routes": routes,
        "center_dev": center_dev,
        # Score balanceado: +éxito -desviación
        "score": (routes * 10) - (center_dev * 50)
    })

df_res = pd.DataFrame(results).sort_values(by="score", ascending=False)
print("--- TOP 5 CHECKPOINTS POR RENDIMIENTO FÍSICO REAL (Town02) ---")
print(df_res.head(5).to_string(index=False))
