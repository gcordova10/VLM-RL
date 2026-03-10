import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"

def get_raw_scalars(path, tag_name):
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    if tag_name not in ea.Tags()['scalars']:
        return []
    return [{'step': e.step, 'value': e.value} for e in ea.Scalars(tag_name)]

print("Extrayendo TODOS los puntos de custom/CPM para Ours...")
raw_data = get_raw_scalars(path_ours, 'custom/CPM')

# Filtrar valores entre 0 y 8
elite_steps = [d['step'] for d in raw_data if 0 <= d['value'] <= 8]

if not elite_steps:
    print("No se encontraron pasos con CPM <= 8 en los datos crudos.")
    # Imprimir una muestra de los valores reales para diagnosticar
    print("\nMuestra de valores reales de CPM encontrados:")
    for d in raw_data[:10]:
        print(f"Step {d['step']}: CPM = {d['value']}")
else:
    print(f"\n✅ Se encontraron {len(elite_steps)} registros que cumplen CPM <= 8.")
    
    # Agrupar en rangos para legibilidad
    ranges = []
    if elite_steps:
        start = elite_steps[0]
        for i in range(1, len(elite_steps)):
            if elite_steps[i] > elite_steps[i-1] + 5000: # Si hay un salto mayor a 5k pasos, cerramos rango
                ranges.append(f"[{start:,} - {elite_steps[i-1]:,}]")
                start = elite_steps[i]
        ranges.append(f"[{start:,} - {elite_steps[-1]:,}]")
    
    print("\nPASOS/EPISODIOS QUE COMPONEN EL OVERLAY:")
    print("\n".join(ranges))

