import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import json
import numpy as np

def get_exact_values(path, target_steps, metrics):
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    results = {int(step): {} for step in target_steps}
    available_tags = ea.Tags()['scalars']
    
    for metric in metrics:
        if metric in available_tags:
            scalars = ea.Scalars(metric)
            steps = np.array([e.step for e in scalars])
            vals = np.array([e.value for e in scalars])
            for target in target_steps:
                results[int(target)][metric] = float(np.interp(target, steps, vals))
        else:
            for target in target_steps:
                results[int(target)][metric] = None
    return results

path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"
path_base = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0"

target_steps = np.linspace(10000, 1000000, 100).astype(int)

# Todas las métricas analizadas
all_metrics = [
    "custom/routes_completed", "custom/total_reward", "custom/mean_reward", "custom/total_distance", "custom/episode_length",
    "custom/avg_speed", "custom/avg_center_dev", "custom/mean_steer_smoothness_x100", 
    "custom/CPM", "custom/CPS", "custom/collision_rate", "custom/collision_num", "custom/collision_speed", "custom/collision_interval",
    "replay_buffer/mean_recent_rewards", "replay_buffer/sum_recent_rewards", "replay_buffer/mean_recent_steer_smoothness_x100",
    "rollout/ep_len_mean", "rollout/ep_gt_rew_mean"
]

print("Extrayendo todas las métricas de Ours...")
data_ours = get_exact_values(path_ours, target_steps, all_metrics)
print("Extrayendo todas las métricas de Baseline...")
data_base = get_exact_values(path_base, target_steps, all_metrics)

comparison_data = []
for step in target_steps:
    comparison_data.append({
        "step": int(step),
        "ours": data_ours[int(step)],
        "base": data_base[int(step)]
    })

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>VLM-RL: Full 100 Checkpoints Comparison</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 20px; }
        .sticky-header { position: sticky; top: 0; background: #0f172a; z-index: 1000; padding: 15px 0; border-bottom: 2px solid #334155; }
        .table-responsive { background: #1e293b; border-radius: 15px; padding: 0; border: 1px solid #334155; overflow-x: auto; max-height: 80vh; }
        .table { color: #e2e8f0; font-size: 0.75rem; white-space: nowrap; margin-bottom: 0; }
        .table th { background: #111827 !important; color: #38bdf8 !important; position: sticky; top: 0; z-index: 10; text-align: center; border-bottom: 2px solid #334155; }
        .table td { border-bottom: 1px solid #334155; padding: 8px 12px; }
        .step-col { position: sticky; left: 0; background: #111827 !important; z-index: 11; font-weight: bold; border-right: 2px solid #334155 !important; }
        .val-ours { color: #f97316; font-weight: bold; }
        .val-base { color: #94a3b8; font-size: 0.7rem; }
        .diff-tag { font-size: 0.65rem; margin-left: 4px; font-weight: bold; }
        h1 { color: #f97316; font-weight: 900; text-transform: uppercase; letter-spacing: 2px; font-size: 1.8rem; }
        .metric-group { background: #0c4a6e !important; color: #fff !important; font-size: 0.7rem; }
    </style>
</head>
<body>
    <div class="sticky-header text-center">
        <h1>📊 Full 100 Checkpoint Audit: Ours vs Baseline</h1>
        <p class="text-secondary small">Comparativa de las 19 métricas | <span style="color:#f97316">Ours</span> vs <span style="color:#94a3b8">Baseline</span> | % de mejora/degradación</p>
    </div>

    <div class="table-responsive shadow-lg">
        <table class="table table-dark table-hover align-middle">
            <thead>
                <tr>
                    <th rowspan="2" class="step-col">Step</th>
                    <th colspan="5" class="metric-group">Success & Progress</th>
                    <th colspan="3" class="metric-group">Control & Precision</th>
                    <th colspan="6" class="metric-group">Safety & Risk</th>
                    <th colspan="3" class="metric-group">Memory & Internal</th>
                    <th colspan="2" class="metric-group">Rollout</th>
                </tr>
                <tr>
                    <!-- Success -->
                    <th>Routes Comp.</th><th>Total Reward</th><th>Mean Reward</th><th>Total Dist.</th><th>Ep. Length</th>
                    <!-- Control -->
                    <th>Avg Speed</th><th>Center Dev</th><th>Smoothness</th>
                    <!-- Safety -->
                    <th>CPM</th><th>CPS</th><th>Coll. Rate</th><th>Coll. Num</th><th>Coll. Speed</th><th>Coll. Interval</th>
                    <!-- Memory -->
                    <th>Buf. Reward</th><th>Buf. Sum</th><th>Buf. Smooth</th>
                    <!-- Rollout -->
                    <th>Roll. Len</th><th>Roll. GT Rew</th>
                </tr>
            </thead>
            <tbody id="table-body"></tbody>
        </table>
    </div>

    <script>
        const data = """ + json.dumps(comparison_data) + """;
        const metrics = """ + json.dumps(all_metrics) + """;
        const tbody = document.getElementById('table-body');

        function format(val) { return (val !== null && !isNaN(val)) ? val.toFixed(3) : '---'; }

        function getDiff(ours, base, metric) {
            if (ours === null || base === null || base === 0) return '';
            const diff = ours - base;
            const pct = (diff / Math.abs(base)) * 100;
            // Métricas donde MENOS es MEJOR
            const inverse = metric.includes('center_dev') || metric.includes('smoothness') || 
                            metric.includes('CPM') || metric.includes('CPS') || 
                            metric.includes('collision_rate') || metric.includes('collision_num') || 
                            metric.includes('collision_speed');
            
            const isBetter = inverse ? diff < 0 : diff > 0;
            const color = isBetter ? '#10b981' : '#ef4444';
            return `<span class="diff-tag" style="color:${color}">${pct > 0 ? '+' : ''}${pct.toFixed(1)}%</span>`;
        }

        data.forEach(d => {
            const tr = document.createElement('tr');
            let html = `<td class="step-col text-center">${d.step.toLocaleString()}</td>`;
            
            metrics.forEach(m => {
                const vO = d.ours[m];
                const vB = d.base[m];
                html += `
                    <td class="text-center">
                        <div class="val-ours">${format(vO)}</div>
                        <div class="val-base">${format(vB)}${getDiff(vO, vB, m)}</div>
                    </td>
                `;
            });
            
            tr.innerHTML = html;
            tbody.appendChild(tr);
        });
    </script>
</body>
</html>
"""

with open("/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard_analysis/detailed_100_comparison.html", "w") as f:
    f.write(html_template)
print("✅ Reporte completo de 100 Checkpoints (19 métricas) generado.")

