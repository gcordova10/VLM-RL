import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import json
import numpy as np

def extract_tb(path):
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    data = {}
    for tag in ea.Tags()['scalars']:
        scalars = ea.Scalars(tag)
        data[tag] = pd.DataFrame([{'step': e.step, 'val': e.value} for e in scalars]).groupby('step')['val'].mean()
    return pd.DataFrame(data)

path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"
path_base = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0"

print("Extrayendo binarios...")
df_ours = extract_tb(path_ours).sort_index().ffill().bfill()
df_base = extract_tb(path_base).sort_index().ffill().bfill()

df_ours.columns = [c + '_ours' for c in df_ours.columns]
df_base.columns = [c + '_base' for c in df_base.columns]

df = df_ours.join(df_base, how='outer').sort_index().ffill().bfill()
df.reset_index(inplace=True)

# Lista de pasos proporcionada por el usuario
marked_steps = [645056, 645248, 645440, 645632, 645824, 646016, 646144, 646336, 646528, 646720, 646912, 647104, 647296, 647488, 647680, 647872, 648064, 648256, 648448, 648640, 648832, 649024, 649216, 649408, 649600, 649792, 649984, 650176, 650368, 650560, 650752, 650944, 651136, 651328, 651520]

filters = [
    "custom/CPM", "custom/CPS", "custom/avg_center_dev", "custom/avg_speed", 
    "custom/collision_interval", "custom/collision_num", "custom/collision_rate", 
    "custom/collision_speed", "custom/episode_length", "custom/mean_reward", 
    "custom/routes_completed", "custom/total_distance", "custom/mean_steer_smoothness_x100", 
    "custom/total_reward", "replay_buffer/mean_recent_rewards", "replay_buffer/sum_recent_rewards", 
    "replay_buffer/mean_recent_steer_smoothness_x100", "rollout/ep_len_mean", "rollout/ep_gt_rew_mean"
]

json_data = df.to_json(orient='records')
steps_json = json.dumps(marked_steps)
metrics_json = json.dumps(filters)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>VLM-RL Elite Highlights</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 20px; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; margin-bottom: 25px; }
        .plot-container { height: 450px; }
        h1 { color: #f97316; font-weight: 900; }
        .highlight-desc { background: #000; padding: 15px; border-radius: 10px; border: 1px solid #38bdf8; color: #38bdf8; font-family: monospace; font-size: 0.8rem; }
    </style>
</head>
<body>
    <h1 class="text-center mb-4">🕵️ ELITE EPISODE HIGHLIGHTS</h1>
    <div class="container mb-5">
        <div class="highlight-desc shadow">
            <b>ZONA DE INTERÉS AUDITADA:</b> Steps 645,056 a 651,520<br>
            Las líneas rojas verticales indican los episodios exactos de máximo rendimiento identificados.
        </div>
    </div>

    <div class="container-fluid">
        <div class="row" id="plots-grid"></div>
    </div>

    <script>
        const data = """ + json_data + """;
        const markedSteps = """ + steps_json + """;
        const metrics = """ + metrics_json + """;
        const plotsGrid = document.getElementById('plots-grid');

        // Generar líneas verticales para todos los gráficos
        const verticalLines = markedSteps.map(s => ({
            type: 'line', xref: 'x', yref: 'paper',
            x0: s, x1: s, y0: 0, y1: 1,
            line: {color: 'rgba(239, 68, 68, 0.4)', width: 1, dash: 'dot'}
        }));

        metrics.forEach((m, idx) => {
            const col = document.createElement('div');
            col.className = 'col-12';
            col.innerHTML = `<div class="card p-3"><h5 class="text-info">${m}</h5><div id="plot-${idx}" class="plot-container"></div></div>`;
            plotsGrid.appendChild(col);

            const keyOurs = m + '_ours';
            const keyBase = m + '_base';
            const hasBase = data.some(d => d[keyBase] !== 0);

            const traces = [{
                x: data.map(d => d.step), y: data.map(d => d[keyOurs]),
                name: 'Ours', line: {color: '#f97316', width: 2}
            }];
            if(hasBase) traces.push({
                x: data.map(d => d.step), y: data.map(d => d[keyBase]),
                name: 'Baseline', line: {color: '#000', width: 1.5}
            });

            Plotly.newPlot('plot-'+idx, traces, {
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(255,255,255,0.05)',
                font: {color: '#94a3b8'},
                xaxis: { title: 'Training Steps', gridcolor: '#334155' },
                yaxis: { title: m, gridcolor: '#334155' },
                shapes: verticalLines,
                margin: {t:30, b:50, l:60, r:20}
            });
        });
    </script>
</body>
</html>
"""

with open("/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard_analysis/nuevo_datos.html", "w") as f:
    f.write(html_template)
print("✅ Reporte de Marcadores Elite generado en tensorboard_analysis/nuevo_datos.html")
