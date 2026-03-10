import argparse
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os

def main():
    # Configuración de argumentos
    parser = argparse.ArgumentParser(description="Exporta todos los scalars de un run de TensorBoard a CSV.")
    parser.add_argument("--logdir", type=str, 
                        default="/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl",
                        help="Ruta absoluta de la carpeta del run de TensorBoard")
    args = parser.parse_args()

    logdir = args.logdir
    if not os.path.exists(logdir):
        print(f"❌ Error: La ruta {logdir} no existe.")
        return

    # Cargar los datos del archivo
    print(f"📂 Cargando datos de: {logdir}...")
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    # Obtener todos los tags de scalars disponibles
    tags = ea.Tags()['scalars']
    if not tags:
        print("⚠️ No se encontraron scalars en este log.")
        return

    print(f"📊 Se encontraron {len(tags)} métricas para exportar.")

    # Definir carpeta de salida: nombre_carpeta_del_run + "_tensorboard_exports"
    folder_name = os.path.basename(os.path.normpath(logdir))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    export_dir = os.path.join(script_dir, f"{folder_name}_tensorboard_exports")
    os.makedirs(export_dir, exist_ok=True)

    print(f"📁 Exportando a: {export_dir}")

    # Exportar cada métrica a un CSV
    for tag in tags:
        try:
            events = ea.Scalars(tag)
            df = pd.DataFrame([(e.step, e.value, e.wall_time) for e in events],
                              columns=['step', 'value', 'wall_time'])
            
            # Limpiar nombre del tag para el nombre del archivo
            clean_tag = tag.replace('/', '_').replace(' ', '_')
            out_path = os.path.join(export_dir, f"{clean_tag}.csv")
            
            df.to_csv(out_path, index=False)
            print(f"  ✅ {tag} -> {len(df)} muestras")
        except Exception as e:
            print(f"  ⚠️ Error exportando {tag}: {e}")

    print(f"\n✨ Exportación completada en: {export_dir}")

if __name__ == "__main__":
    main()