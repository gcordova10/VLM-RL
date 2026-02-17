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
        # Forzamos nombres únicos desde la extracción
        data[tag] = pd.DataFrame([{'step': e.step, 'val': e.value} for e in scalars]).groupby('step')['val'].mean()
    return pd.DataFrame(data)

path_base = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0"
path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"

print("Extrayendo binarios y unificando claves...")
df_base = extract_tb(path_base)
df_ours = extract_tb(path_ours)

# Forzamos sufijos manualmente para evitar que el join los omita si no hay colisión
df_base.columns = [c + '_base' for c in df_base.columns]
df_ours.columns = [c + '_ours' for c in df_ours.columns]

# Unir todo por pasos de entrenamiento
df = df_ours.join(df_base, how='outer').sort_index().interpolate(method='linear')
df.reset_index(inplace=True)

filters = [
    {"metric": "custom/CPM", "op": "<", "val": 10},
    {"metric": "custom/CPS", "op": "<", "val": 0.01},
    {"metric": "custom/avg_center_dev", "op": "<", "val": 0.2},
    {"metric": "custom/avg_speed", "op": ">", "val": 15},
    {"metric": "custom/collision_interval", "op": ">", "val": 12000},
    {"metric": "custom/collision_num", "op": "<", "val": 5},
    {"metric": "custom/collision_rate", "op": "<", "val": 0.5},
    {"metric": "custom/collision_speed", "op": "<", "val": 4},
    {"metric": "custom/episode_length", "op": ">", "val": 5000},
    {"metric": "custom/mean_reward", "op": ">", "val": 0.5},
    {"metric": "custom/routes_completed", "op": ">", "val": 5},
    {"metric": "custom/total_distance", "op": ">", "val": 2000},
    {"metric": "custom/mean_steer_smoothness_x100", "op": "<", "val": 15},
    {"metric": "custom/total_reward", "op": ">", "val": 2000},
    {"metric": "replay_buffer/mean_recent_rewards", "op": ">", "val": 0.5},
    {"metric": "replay_buffer/sum_recent_rewards", "op": ">", "val": 200},
    {"metric": "replay_buffer/mean_recent_steer_smoothness_x100", "op": "<", "val": 8},
    {"metric": "rollout/ep_len_mean", "op": ">", "val": 4000},
    {"metric": "rollout/ep_gt_rew_mean", "op": ">", "val": 100}
]

json_data = df.to_json(orient='records')

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>VLM-RL Dual Audit Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 30px; }}
        .card {{ background: #1e293b; border: 1px solid #334155; margin-bottom: 30px; border-radius: 15px; overflow: hidden; }}
        .filter-ui {{ padding: 15px; background: #111827; border-bottom: 1px solid #334155; }}
        .metric-label {{ color: #f97316; font-weight: 800; font-size: 0.9rem; }}
        .badge-val {{ background: #0f172a; border: 1px solid #ef4444; color: #ef4444; padding: 5px 10px; border-radius: 8px; font-family: monospace; text-align: center; }}
        .plot-container {{ height: 400px; }}
        h1 {{ color: #f97316; font-weight: 900; text-transform: uppercase; letter-spacing: 2px; }}
        input[type=range] {{ width: 100%; cursor: pointer; }}
        .legend-box {{ font-size: 0.75rem; color: #94a3b8; }}
    </style>
</head>
<body>
    <h1 class="text-center mb-5">🕵️ VLM-RL Dual Audit</h1>
    <div class="container-fluid">
        <div class="row" id="plots-grid"></div>
    </div>

    <script>
        const data = {json_data};
        const filterConfigs = {json.dumps(filters)};
        const plotsGrid = document.getElementById('plots-grid');

        filterConfigs.forEach((conf, idx) => {{
            const keyOurs = conf.metric + '_ours';
            const keyBase = conf.metric + '_base';
            
            // Verificar si hay datos reales para cada uno
            const hasOurs = data.some(d => d[keyOurs] !== null);
            const hasBase = data.some(d => d[keyBase] !== null);

            if (!hasOurs && !hasBase) return; // Saltamos si no hay nada

            const col = document.createElement('div');
            col.className = 'col-xl-6 col-lg-12';
            col.innerHTML = `
                <div class="card">
                    <div class="filter-ui">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="metric-label">${{conf.metric}}</span>
                            <div class="d-flex align-items-center gap-3">
                                <div class="legend-box">
                                    <span style="display:inline-block; width:10px; height:10px; background:rgba(249,115,22,0.3); border:1px solid #f97316"></span> Ours
                                    ${{hasBase ? '<span class="ms-2" style="display:inline-block; width:10px; height:10px; background:rgba(255,255,255,0.15); border:1px solid #fff"></span> Base' : ''}}
                                </div>
                                <input type="number" id="input-${{idx}}" class="badge-val" style="width: 100px;" step="any">
                            </div>
                        </div>
                        <input type="range" id="slider-${{idx}}" step="any">
                    </div>
                    <div id="plot-${{idx}}" class="plot-container"></div>
                </div>
            `;
            plotsGrid.appendChild(col);

            const traces = [];
            if(hasOurs) traces.push({{ x: data.map(d => d.step), y: data.map(d => d[keyOurs]), name: 'Ours', type: 'scatter', line: {{color: '#f97316', width: 2.5}} }});
            if(hasBase) traces.push({{ x: data.map(d => d.step), y: data.map(d => d[keyBase]), name: 'Baseline', type: 'scatter', line: {{color: '#000000', width: 2}} }});

            const layout = {{
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(255,255,255,0.05)',
                font: {{color: '#94a3b8'}}, margin: {{t:20, b:40, l:60, r:20}},
                xaxis: {{gridcolor: '#334155', zeroline: false}},
                yaxis: {{gridcolor: '#334155', zeroline: false}},
                legend: {{orientation: "h", y: 1.1}},
                shapes: []
            }};

            Plotly.newPlot(`plot-${{idx}}`, traces, layout);

            const slider = document.getElementById(`slider-${{idx}}`);
            const input = document.getElementById(`input-${{idx}}`);
            
            const allVals = data.map(d => d[keyOurs]).concat(data.map(d => d[keyBase])).filter(v => v !== null);
            const min = Math.min(...allVals);
            const max = Math.max(...allVals);
            slider.min = min; slider.max = max; slider.value = conf.val; input.value = conf.val;

            const update = (source) => {{
                let thresh = source === 'slider' ? parseFloat(slider.value) : parseFloat(input.value);
                if(source === 'slider') input.value = thresh.toFixed(4); else slider.value = thresh;
                
                const shapes = [{{
                    type: 'line', xref: 'paper', x0: 0, x1: 1,
                    yref: 'y', y0: thresh, y1: thresh,
                    line: {{color: '#ef4444', width: 3}}
                }}];

                const generateBoundedRects = (key, color) => {{
                    let start = null;
                    for (let i = 0; i < data.length; i++) {{
                        const val = data[i][key];
                        if (val === null) continue;
                        const condition = conf.op === '<' ? val < thresh : val > thresh;
                        if (condition) {{
                            if (start === null) start = data[i].step;
                        }} else if (start !== null) {{
                            shapes.push({{
                                type: 'rect', xref: 'x', yref: 'y',
                                x0: start, x1: data[i-1].step,
                                y0: conf.op === '<' ? 0 : thresh,
                                y1: conf.op === '<' ? thresh : max * 1.5,
                                fillcolor: color, line: {{width: 0}}
                            }});
                            start = null;
                        }}
                    }}
                    if (start !== null) {{
                        shapes.push({{
                            type: 'rect', xref: 'x', yref: 'y',
                            x0: start, x1: data[data.length-1].step,
                            y0: conf.op === '<' ? 0 : thresh,
                            y1: conf.op === '<' ? thresh : max * 1.5,
                            fillcolor: color, line: {{width: 0}}
                        }});
                    }}
                }};

                if(hasOurs) generateBoundedRects(keyOurs, 'rgba(249, 115, 22, 0.25)');
                if(hasBase) generateBoundedRects(keyBase, 'rgba(255, 255, 255, 0.15)');

                Plotly.relayout(`plot-${{idx}}`, {{shapes: shapes}});
            }};

            slider.oninput = () => update('slider');
            input.onchange = () => update('input');
            update('slider');
        }});
    </script>
</body>
</html>
"""

output_path = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard_analysis/comparison_dashboard_nuevo.html"
with open(output_path, "w") as f:
    f.write(html_content)
print("✅ Dashboard corregido: Cargando métricas de suavidad exclusivas.")
