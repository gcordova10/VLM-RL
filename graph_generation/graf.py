#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# === CONFIGURACIÓN ===
BASE_DIR = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard"
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === ESTILO GRÁFICO ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
})

# Colores y estilos consistentes
RUN_STYLES = {
    "VLM–RL (baseline)": {"color": "black", "linestyle": "-", "linewidth": 2.0, "marker": None},
    "CLG–Smooth (ours)": {"color": "#00bcd4", "linestyle": "-", "linewidth": 2.0, "marker": None},
}

# Asocia carpetas a etiquetas legibles
RUN_MAP = {
    "CLIPRewardedSAC_20250930_154046_idvlm_rl": "VLM–RL (baseline)",
    "CLIPRewardedSAC_20251027_081939_idvlm_rl": "CLG–Smooth (ours)",
}

# Métricas
METRICS = [
    ("custom/CPM", "CPM", 999_228),
    ("custom/collision_rate", "Collision Rate", 999_228),
    ("custom/routes_completed", "Routes Completed", 999_228),
    ("train/ent_coef", "Entropy Coefficient", 1_000_000),
]

# Límites personalizados
YLIM_CUSTOM = {
    "custom/CPM": (0, 200),
    "train/ent_coef": (0, 0.02),
}


def find_tfevents_file(run_dir):
    """Busca el archivo .tfevents dentro de la carpeta del run"""
    for f in os.listdir(run_dir):
        if f.startswith("events.out.tfevents"):
            return os.path.join(run_dir, f)
    return None


def load_df(path, tag):
    """Carga los datos de TensorBoard"""
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        print(f"⚠️  '{tag}' no encontrado en {path}")
        return None
    events = ea.Scalars(tag)
    df = pd.DataFrame([(e.step, e.value) for e in events], columns=["step", "value"])
    df["smooth"] = df["value"].ewm(alpha=0.1).mean()
    return df


def plot_metric(ax, tag, ylabel, ref_step, xlabel_size=12, ylabel_size=13, include_legend=True):
    """Dibuja cada métrica con estilo limpio"""
    for folder, label in RUN_MAP.items():
        run_dir = os.path.join(BASE_DIR, folder)
        tfevents_path = find_tfevents_file(run_dir)
        if not tfevents_path:
            print(f"❌ No se encontró .tfevents en {run_dir}")
            continue

        df = load_df(tfevents_path, tag)
        if df is None or df.empty:
            continue

        style = RUN_STYLES[label]
        ax.plot(
            df["step"],
            df["smooth"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=label if include_legend else None,
        )

        # Línea vertical de referencia
        ax.axvline(ref_step, color="gray", linestyle="--", linewidth=1)

    # Ejes y formato
    ax.set_xlabel("Training Steps", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    if tag in YLIM_CUSTOM:
        ax.set_ylim(*YLIM_CUSTOM[tag])

    # === Leyenda dentro del gráfico ===
    if include_legend:
        ax.legend(
            loc="upper left",
            fontsize=11,
            frameon=True,
            facecolor="white",
            framealpha=0.8,
            edgecolor="gray",
        )


def save_combined_metrics():
    """Genera la figura combinada con las 4 métricas (leyendas internas)"""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    plt.subplots_adjust(wspace=0.3, top=0.90)

    for ax, (tag, ylabel, ref_step) in zip(axes, METRICS):
        plot_metric(ax, tag, ylabel, ref_step, include_legend=True)

    out_file = os.path.join(OUTPUT_DIR, "all_metrics_inside_legend_600dpi.png")
    plt.savefig(out_file, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figura combinada guardada: {out_file}")


def save_individual_metrics():
    """Genera una imagen separada para cada métrica (sin leyenda y texto más grande)"""
    for tag, ylabel, ref_step in METRICS:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_metric(ax, tag, ylabel, ref_step, xlabel_size=15, ylabel_size=16, include_legend=True)

        tag_name = tag.split("/")[-1]
        out_file = os.path.join(OUTPUT_DIR, f"{tag_name}_600dpi.png")
        plt.savefig(out_file, dpi=600, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Figura individual guardada: {out_file}")


def main():
    save_combined_metrics()
    save_individual_metrics()


if __name__ == "__main__":
    main()
