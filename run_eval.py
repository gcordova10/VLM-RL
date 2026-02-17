import glob
import subprocess
import time
import os

from tqdm import tqdm


def kill_carla():
    """Cierra cualquier instancia de CARLA en ejecución."""
    print("Killing Carla server...\n")
    time.sleep(1)
    subprocess.run(
        ["killall", "-9", "CarlaUE4-Linux-Shipping"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(4)


def format_duration(seconds):
    """Convierte segundos a formato mm:ss"""
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"


if __name__ == '__main__':
    tensorboard_path = './tensorboard'

    # Ciudades (towns) y densidades
    towns = [f"Town0{i}" for i in range(1, 6)]  # Town01 ... Town05
    densities = ["empty", "regular", "dense"]
    port = 2020

    # Recorre todas las carpetas de modelos dentro de tensorboard/
    for model_path in tqdm(sorted(os.listdir(tensorboard_path)), desc="Processing model folders"):
        full_model_path = os.path.join(tensorboard_path, model_path)
        if not os.path.isdir(full_model_path):
            continue

        # Detecta automáticamente la configuración (como antes)
        config = model_path.split('id')[-1]
        print("\n" + "=" * 60)
        print(f"Evaluating model folder: {model_path}")
        print(f"Config detected: {config}")
        print("=" * 60)

        # Encuentra todos los checkpoints .zip en la carpeta del model_path
        model_ckpts = sorted(glob.glob(os.path.join(full_model_path, "model_*_steps.zip")))
        if not model_ckpts:
            print(f"No checkpoints found in {model_path}, skipping...\n")
            continue

        # Itera sobre cada checkpoint
        for model_ckpt in model_ckpts:
            model_name = os.path.basename(model_ckpt)
            print(f"\n>>> Starting evaluation for checkpoint: {model_name}\n")

            start_time = time.time()
            # Ejecuta en todas las ciudades y densidades
            for town in towns:
                for density in densities:
                    print(f"\n=== Running {model_name} | {town} | {density} ===")
                    kill_carla()

                    args_eval = [
                        "--model", model_ckpt,
                        "--config", config,
                        "--town", town,
                        "--density", density,
                        "--port", str(port)
                    ]

                    try:
                        subprocess.run(["python", "eval.py"] + args_eval, check=True)
                    except subprocess.CalledProcessError:
                        print(f"⚠️ Error evaluating {model_name} | {town} | {density}, skipping...\n")
                        continue
                    finally:
                        port += 2

                    print(f"✅ Finished {town} - {density}\n")
                    time.sleep(5)

            duration = time.time() - start_time
            print(f"✅ Finished checkpoint {model_name} | Duration: {format_duration(duration)}")
            print("=" * 60)

    print("\n🎯 Evaluación completada para todas las combinaciones de checkpoints, towns y densities.\n")
