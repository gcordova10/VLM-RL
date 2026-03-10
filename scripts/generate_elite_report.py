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

path_base = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0"
path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"

print("Extrayendo datos crudos (sin ffill)...")
df_base = extract_tb(path_base)
df_ours = extract_tb(path_ours)

df_base.columns = [c + '_base' for c in df_base.columns]
df_ours.columns = [c + '_ours' for c in df_ours.columns]

# Unimos tal cual, manteniendo los nulos donde no haya datos grabados
df = df_ours.join(df_base, how='outer').sort_index()
df.reset_index(inplace=True)

filters = [
    {"metric": "custom/CPM", "op": "<", "val": 8},
    {"metric": "custom/CPS", "op": "<", "val": 0.005},
    {"metric": "custom/avg_center_dev", "op": "<", "val": 0.1},
    {"metric": "custom/avg_speed", "op": ">", "val": 15},
    {"metric": "custom/collision_interval", "op": ">", "val": 5000},
    {"metric": "custom/collision_num", "op": "<", "val": 230},
    {"metric": "custom/collision_rate", "op": "<", "val": 0.8},
    {"metric": "custom/collision_speed", "op": "<", "val": 4},
    {"metric": "custom/episode_length", "op": ">", "val": 3000},
    {"metric": "custom/mean_reward", "op": ">", "val": 0.5},
    {"metric": "custom/routes_completed", "op": ">", "val": 4},
    {"metric": "custom/total_distance", "op": ">", "val": 2000},
    {"metric": "custom/mean_steer_smoothness_x100", "op": "<", "val": 15},
    {"metric": "custom/total_reward", "op": ">", "val": 2000},
    {"metric": "replay_buffer/mean_recent_rewards", "op": ">", "val": 0.5},
    {"metric": "replay_buffer/sum_recent_rewards", "op": ">", "val": 200},
    {"metric": "replay_buffer/mean_recent_steer_smoothness_x100", "op": "<", "val": 8},
    {"metric": "rollout/ep_len_mean", "op": ">", "val": 3000},
    {"metric": "rollout/ep_gt_rew_mean", "op": ">", "val": 100}
]

json_data = df.to_json(orient='records')
filters_json = json.dumps(filters)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>VLM-RL Elite Finder (RAW)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 0; }
        .header-section { position: sticky; top: 0; z-index: 1000; background: #0f172a; padding: 20px; border-bottom: 2px solid #334155; }
        .elite-box { background: #000; border: 1px solid #334155; height: 180px; overflow-y: auto; color: #38bdf8; font-family: monospace; padding: 10px; font-size: 0.8rem; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; margin-bottom: 20px; overflow: hidden; }
        .filter-ui { padding: 12px; background: #111827; border-bottom: 1px solid #334155; }
        .badge-val { background: #0f172a; border: 1px solid #ef4444; color: #ef4444; width: 90px; text-align: center; }
        .plot-container { height: 350px; }
    </style>
</head>
<body>
    <div class="header-section text-center">
        <h1 style="color:#f97316; font-weight:900;">🕵️ RAW ELITE FINDER</h1>
        <div class="container-fluid mt-3">
            <div class="row">
                <div class="col-6 border-end border-secondary">
                    <div class="small text-white mb-1">🏆 BASELINE (17 criteria)</div>
                    <div id="baseline-elite" class="elite-box rounded"></div>
                </div>
                <div class="col-6">
                    <div class="small mb-1" style="color:#f97316;">🔥 OURS (19 criteria)</div>
                    <div id="ours-elite" class="elite-box rounded" style="border-color:#f97316; color:#f97316;"></div>
                </div>
            </div>
        </div>
    </div>

    <div class="container-fluid p-4">
        <div class="row" id="plots-grid"></div>
    </div>

    <script>
        const data = """ + json_data + """;
        const filterConfigs = """ + filters_json + """;
        const plotsGrid = document.getElementById('plots-grid');

        filterConfigs.forEach((conf, idx) => {
            const col = document.createElement('div');
            col.className = 'col-md-6';
            const keyOurs = conf.metric + '_ours';
            const keyBase = conf.metric + '_base';
            const hasOurs = data.some(d => d[keyOurs] !== null);
            const hasBase = data.some(d => d[keyBase] !== null);
            if (!hasOurs && !hasBase) return;

            col.innerHTML = `
                <div class="card shadow-sm">
                    <div class="filter-ui">
                        <div class="d-flex justify-content-between align-items-center">
                            <span style="color:#f97316; font-weight:bold">${conf.metric}</span>
                            <input type="number" id="input-${idx}" class="badge-val" step="any">
                        </div>
                        <input type="range" id="slider-${idx}" step="any" class="w-100 mt-2">
                    </div>
                    <div id="plot-${idx}" class="plot-container"></div>
                </div>
            `;
            plotsGrid.appendChild(col);
            renderPlot(idx, conf, hasBase, hasOurs);
        });

        function calculateElite() {
            const current = filterConfigs.map((c, i) => {
                const el = document.getElementById(`input-${i}`);
                return { metric: c.metric, val: el ? parseFloat(el.value) : c.val, op: c.op };
            });

            const findSteps = (suffix) => data.filter(d => {
                return current.every(t => {
                    const v = d[t.metric + suffix];
                    if (suffix === '_base' && t.metric.includes('smoothness')) return true;
                    // MUY ESTRICTO: Si no hay valor grabado en ese step, no cumple
                    if (v === null) return false; 
                    return t.op === '<' ? v < t.val : v > t.val;
                });
            }).map(d => d.step);

            const bList = findSteps('_base');
            const oList = findSteps('_ours');

            document.getElementById('baseline-elite').innerHTML = bList.length + ' pasos:<br>' + bList.join(' | ');
            document.getElementById('ours-elite').innerHTML = oList.length + ' pasos:<br>' + oList.join(' | ');
        }

        function renderPlot(idx, conf, hasBase, hasOurs) {
            const keyOurs = conf.metric + '_ours';
            const keyBase = conf.metric + '_base';
            const traces = [];
            if(hasOurs) traces.push({ x: data.map(d => d.step), y: data.map(d => d[keyOurs]), name: 'Ours', line: {color: '#f97316', width: 2} });
            if(hasBase) traces.push({ x: data.map(d => d.step), y: data.map(d => d[keyBase]), name: 'Base', line: {color: '#000000', width: 1.5} });

            Plotly.newPlot(`plot-${idx}`, traces, {
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(255,255,255,0.05)',
                font: {color: '#94a3b8'}, margin: {t:10, b:30, l:50, r:10},
                xaxis: {gridcolor: '#334155'}, yaxis: {gridcolor: '#334155'},
                legend: {orientation: 'h', y: 1.1}
            });

            const slider = document.getElementById(`slider-${idx}`);
            const input = document.getElementById(`input-${idx}`);
            const all = data.map(d => d[keyOurs]).concat(data.map(d => d[keyBase])).filter(v => v !== null);
            slider.min = Math.min(...all); slider.max = Math.max(...all); slider.value = conf.val; input.value = conf.val;

            const update = () => {
                let thresh = parseFloat(input.value); slider.value = thresh;
                const shapes = [{ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: thresh, y1: thresh, line: {color: '#ef4444', width: 2} }];
                Plotly.relayout(`plot-${idx}`, {shapes: shapes});
                calculateElite();
            };
            slider.oninput = () => { input.value = parseFloat(slider.value).toFixed(4); update(); };
            input.onchange = update;
            update();
        }
    </script>
</body>
</html>
"""

with open("/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard_analysis/elite_finder.html", "w") as f:
    f.write(html_template)
print("✅ Buscador Elite RAW generado (Sin ffill, datos estrictos).")
