from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os

# Ruta del archivo .tfevents
logdir = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl"

# Cargar los datos del archivo
ea = event_accumulator.EventAccumulator(logdir)
ea.Reload()

print("Scalars disponibles:")
for tag in ea.Tags()['scalars']:
    print("  ", tag)

# Elegir las métricas que quieres exportar (puedes agregar más)
tags = [
    'custom/collision_rate',
    'custom/CPM',
    'train/ent_coef',
    'custom/routes_completed'
]

# Crear una carpeta de salida relativa al script
script_dir = os.path.dirname(os.path.abspath(__file__))
export_dir = os.path.join(script_dir, "tensorboard_exports")
os.makedirs(export_dir, exist_ok=True)

# Exportar cada métrica a un CSV
for tag in tags:
    try:
        events = ea.Scalars(tag)
        df = pd.DataFrame([(e.step, e.value, e.wall_time) for e in events],
                          columns=['step', 'value', 'wall_time'])
        out_path = os.path.join(export_dir, f"{tag.replace('/', '_')}.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ Guardado: {out_path} ({len(df)} muestras)")
    except KeyError:
        print(f"⚠️ No encontrada: {tag}")
