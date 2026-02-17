import os
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
BASE_DIR = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard"
TAG = "train/ent_coef"
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REFERENCE_STEP = 1_000_000  # paso de referencia mostrado en TensorBoard

RUN_COLORS = {
    "CLIPRewardedSAC_20250930_154046_idvlm_rl": "#1f1f1f",  # gris oscuro
    "CLIPRewardedSAC_20251027_081939_idvlm_rl": "#00bcd4",  # celeste
}


def find_tfevents_file(run_dir):
    """Busca el archivo .tfevents dentro de la carpeta del run"""
    for f in os.listdir(run_dir):
        if f.startswith("events.out.tfevents"):
            return os.path.join(run_dir, f)
    return None


def load_df(path, tag):
    """Carga los datos desde un archivo .tfevents"""
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        print(f"⚠️  '{tag}' no encontrado en {path}")
        return None
    events = ea.Scalars(tag)
    df = pd.DataFrame([(e.step, e.value) for e in events], columns=["step", "value"])
    df["smooth"] = df["value"].ewm(alpha=0.1).mean()
    return df


def get_value_at_step(df, step):
    """Obtiene el valor y valor suavizado más cercano al paso indicado"""
    closest_idx = (df["step"] - step).abs().idxmin()
    row = df.iloc[closest_idx]
    return float(row["value"]), float(row["smooth"]), int(row["step"])


def main():
    plt.figure(figsize=(10, 6))
    legend_texts = []

    for run_name, color in RUN_COLORS.items():
        run_dir = os.path.join(BASE_DIR, run_name)
        tfevents_path = find_tfevents_file(run_dir)
        if not tfevents_path:
            print(f"❌ No se encontró .tfevents en {run_dir}")
            continue

        print(f"📂 Procesando: {tfevents_path}")
        df = load_df(tfevents_path, TAG)
        if df is None or df.empty:
            continue

        plt.plot(df["step"], df["smooth"], color=color, linewidth=2)

        value, smoothed, real_step = get_value_at_step(df, REFERENCE_STEP)
        legend_texts.append(
            f"{run_name}   Smoothed={smoothed:.4f}   Value={value:.4f}   (step={real_step})"
        )

    if not legend_texts:
        print("⚠️ No se encontró la métrica en ningún run.")
        return

    plt.title(TAG)
    plt.xlabel("Training Steps")
    plt.ylabel("Entropy Coefficient")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(legend_texts, loc="upper right", frameon=False, fontsize=9)

    out_file = os.path.join(OUTPUT_DIR, "ent_coef_comparison_refstep.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, facecolor="white")
    plt.show()
    print(f"✅ Gráfico guardado en: {out_file}")


if __name__ == "__main__":
    main()
