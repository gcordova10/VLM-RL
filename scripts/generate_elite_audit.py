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

print("Extrayendo binarios...")
df_base = extract_tb(path_base).sort_index().ffill().bfill()
df_ours = extract_tb(path_ours).sort_index().ffill().bfill()

df_base.columns = [c + '_base' for c in df_base.columns]
df_ours.columns = [c + '_ours' for c in df_ours.columns]

df = df_ours.join(df_base, how='outer').sort_index().ffill().bfill()
df.reset_index(inplace=True)

if len(df) > 5000:
    df = df.iloc[::len(df)//5000]

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
    <title>VLM-RL Elite Intersection</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 0; }
        .sticky-top { position: sticky; top: 0; z-index: 1000; background: #0f172a; padding: 20px; border-bottom: 2px solid #334155; }
        .elite-box { background: #000; border: 1px solid #334155; height: 180px; overflow-y: auto; color: #38bdf8; font-family: monospace; padding: 10px; font-size: 0.8rem; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; margin-bottom: 20px; overflow: hidden; }
        .badge-val { background: #0f172a; border: 1px solid #ef4444; color: #ef4444; width: 90px; text-align: center; }
        .plot-container { height: 350px; }
    </style>
</head>
<body>
    <div class="sticky-top shadow-lg">
        <h1 class="text-center mb-3" style="color:#f97316; font-weight:900;">🕵️ ELITE INTERSECTION AUDITOR</h1>
        <div class="container-fluid">
            <div class="row">
                <div class="col-6 border-end border-secondary">
                    <div class="text-center text-white small mb-1">🏆 BASELINE ELITE (17 criteria)</div>
                    <div id="baseline-elite" class="elite-box rounded"></div>
                </div>
                <div class="col-6">
                    <div class="text-center small mb-1" style="color:#f97316;">🔥 OURS ELITE (19 criteria)</div>
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
            const col = document.createElement('div'); col.className = 'col-xl-6 col-lg-12';
            const keyBase = conf.metric + '_base';
            const hasBase = data.some(d => d[keyBase] !== 0);
            col.innerHTML = `
                <div class="card p-3 shadow-sm">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <span style="color:#f97316; font-weight:bold">${conf.metric}</span>
                        <input type="number" id="input-${idx}" class="badge-val" value="${conf.val}" step="any" oninput="updateAll()">
                    </div>
                    <input type="range" id="slider-${idx}" step="any" class="w-100" oninput="sync(${idx})">
                    <div id="plot-${idx}" class="plot-container"></div>
                </div>`;
            plotsGrid.appendChild(col);
            initPlot(idx, conf, hasBase);
        });

        function initPlot(idx, conf, hasBase) {
            const traces = [{ x: data.map(d => d.step), y: data.map(d => d[conf.metric+'_ours']), name: 'Ours', line: {color:'#f97316', width:2} }];
            if(hasBase) traces.push({ x: data.map(d => d.step), y: data.map(d => d[conf.metric+'_base']), name: 'Base', line: {color:'#000', width:1.5} });
            Plotly.newPlot('plot-'+idx, traces, {
                paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(255,255,255,0.05)',
                font:{color:'#94a3b8'}, margin:{t:10, b:30, l:50, r:10},
                xaxis:{gridcolor:'#334155'}, yaxis:{gridcolor:'#334155'},
                legend:{orientation:'h', y:1.1}, shapes:[]
            });
            const s = document.getElementById('slider-'+idx);
            const all = data.map(d => d[conf.metric+'_ours']).concat(data.map(d => d[conf.metric+'_base'])).filter(v => v!==0);
            s.min = Math.min(...all); s.max = Math.max(...all); s.value = conf.val;
        }

        function sync(i) {
            const s = document.getElementById('slider-'+i);
            const input = document.getElementById('input-'+i);
            input.value = parseFloat(s.value).toFixed(4);
            updateAll();
        }

        function updateAll() {
            const current = filterConfigs.map((c, i) => ({
                metric: c.metric, val: parseFloat(document.getElementById('input-'+i).value), op: c.op
            }));

            const find = (suffix) => data.filter(d => {
                return current.every(t => {
                    if (suffix === '_base' && t.metric.includes('smoothness')) return true;
                    const v = d[t.metric + suffix];
                    if (v === null || v === 0) return false;
                    return t.op === '<' ? v < t.val : v > t.val;
                });
            }).map(d => d.step);

            const bList = find('_base'); const oList = find('_ours');
            document.getElementById('baseline-elite').innerHTML = bList.length + ' pasos:<br>' + bList.join(' | ');
            document.getElementById('ours-elite').innerHTML = oList.length + ' pasos:<br>' + oList.join(' | ');

            filterConfigs.forEach((conf, idx) => {
                const thresh = parseFloat(document.getElementById('input-'+idx).value);
                const shapes = [{ type:'line', xref:'paper', x0:0, x1:1, yref:'y', y0:thresh, y1:thresh, line:{color:'#ef4444', width:2} }];
                
                const draw = (key, color) => {
                    let s = null;
                    for(let i=0; i<data.length; i++) {
                        const v = data[i][key]; if(v===null) continue;
                        const cond = conf.op === '<' ? v < thresh : v > thresh;
                        if(cond) { if(s===null) s=data[i].step; }
                        else if(s!==null) {
                            shapes.push({ type:'rect', xref:'x', yref:'y', x0:s, x1:data[i-1].step, y0:conf.op==='<'?0:thresh, y1:conf.op==='<'?thresh:thresh*2, fillcolor:color, line:{width:0} });
                            s = null;
                        }
                    }
                };
                draw(conf.metric+'_ours', 'rgba(249,115,22,0.2)');
                if(data.some(d => d[conf.metric+'_base'] !== 0)) draw(conf.metric+'_base', 'rgba(255,255,255,0.1)');
                Plotly.relayout('plot-'+idx, {shapes: shapes});
            });
        }
        window.onload = updateAll;
    </script>
</body>
</html>
"""

with open("/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard_analysis/elite_finder.html", "w") as f:
    f.write(html_template)
print("✅ Buscador Elite Definitivo generado en tensorboard_analysis/elite_finder.html")
