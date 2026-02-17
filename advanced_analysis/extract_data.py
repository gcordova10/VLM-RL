import os
import pandas as pd
import json
import re

def extract_step(filename):
    match = re.search(r'model_(\d+)_steps', filename)
    if match:
        return int(match.group(1))
    return None

base_path = '/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl'
eval_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith('eval')]

data = {}

for eval_dir in eval_dirs:
    dir_path = os.path.join(base_path, eval_dir)
    if not os.path.isdir(dir_path):
        continue
    summary_files = [f for f in os.listdir(dir_path) if f.endswith('_summary.csv')]
    
    for f in summary_files:
        step = extract_step(f)
        if step is None:
            continue
            
        file_path = os.path.join(dir_path, f)
        try:
            df = pd.read_csv(file_path)
            if 'total' in df['episode'].values:
                total_row = df[df['episode'] == 'total'].iloc[0]
            else:
                total_row = df.iloc[-1]
            
            # Extraer las 5 métricas clave
            as_v = float(total_row['speed_mean'])
            rc_v = float(total_row['routes_completed'])
            td_v = float(total_row['total_distance'])
            sr_v = float(total_row['success'])
            cs_v = float(total_row['collision_speed'])
            
            # Guardar datos brutos para normalización posterior
            if step not in data:
                data[step] = {}
            
            data[step][eval_dir] = {
                'as': as_v, 'rc': rc_v, 'td': td_v, 'sr': sr_v, 'cs': cs_v
            }
        except Exception as e:
            continue

# --- PROCESAMIENTO Y NORMALIZACIÓN ---
steps = sorted(data.keys())
scenarios = sorted(eval_dirs)

# Encontrar máximos globales para normalizar
max_vals = {'as': 0.1, 'td': 0.1, 'cs': 0.1}
for s_data in data.values():
    for scen_data in s_data.values():
        for k in max_vals:
            if scen_data[k] > max_vals[k]: max_vals[k] = scen_data[k]

heatmap_data = []
stats = []

for step in steps:
    row = {'step': step}
    upi_values = []
    for scenario in scenarios:
        d = data[step].get(scenario, None)
        if d:
            # Calcular UPI (0.0 a 1.0)
            # RC y SR ya son 0-1. AS, TD y CS se normalizan.
            norm_as = d['as'] / max_vals['as']
            norm_td = d['td'] / max_vals['td']
            norm_safe = 1.0 - (d['cs'] / max_vals['cs']) # Inverso: menos choque es mejor
            
            upi = (norm_as + d['rc'] + norm_td + d['sr'] + norm_safe) / 5.0
            upi = max(0, min(1.0, upi)) # Asegurar rango
        else:
            upi = 0
            
        row[scenario] = upi
        upi_values.append(upi)
    
    avg_upi = sum(upi_values) / len(upi_values) if upi_values else 0
    variance = sum((x - avg_upi)**2 for x in upi_values) / len(upi_values) if upi_values else 0
    stability = 1.0 - (variance ** 0.5)
    
    stats.append({
        'step': step,
        'avg_success': avg_upi, # Mantenemos nombre de variable por compatibilidad con template
        'stability': stability
    })
    heatmap_data.append(row)

best_by_avg = sorted(stats, key=lambda x: x['avg_success'], reverse=True)[:10]
best_by_stability = sorted(stats, key=lambda x: x['stability'], reverse=True)[:10]

output = {
    'steps': steps,
    'scenarios': [s.replace('eval', '') for s in scenarios],
    'heatmap': heatmap_data,
    'best_models': {
        'by_avg_success': best_by_avg,
        'by_stability': best_by_stability
    }
}

json_data_str = json.dumps(output, indent=4)

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>VLM-RL Advanced UPI Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }
        .card { background: #1e293b; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px; border: 1px solid #334155; }
        h1 { color: #38bdf8; text-align: center; font-weight: 800; text-transform: uppercase; letter-spacing: 2px; }
        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
        .stat-card { padding: 15px; border-radius: 8px; border-left: 5px solid; }
        .blue { background: #0c4a6e; border-color: #38bdf8; }
        .green { background: #064e3b; border-color: #10b981; }
        .orange { background: #451a03; border-color: #f59e0b; }
        ul { padding-left: 20px; }
        .info-bar { background: #334155; padding: 10px; border-radius: 8px; font-size: 0.9rem; margin-bottom: 20px; text-align: center; }
    </style>
</head>
<body>
    <h1>🚀 VLM-RL: UPI Performance Analysis</h1>
    
    <div class="card">
        <h3>📐 Metodología de Cálculo (UPI)</h3>
        <p>El <b>Unified Performance Index</b> es una métrica compuesta que equilibra eficiencia, progreso y seguridad:</p>
        <div style="background: #0f172a; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.95rem; border: 1px solid #38bdf8;">
            UPI = [ (AS / maxAS) + RC + (TD / maxTD) + SR + (1 - CS / maxCS) ] / 5 * 100
        </div>
        <ul style="margin-top: 10px; font-size: 0.85rem; color: #94a3b8;">
            <li><b>AS (Average Speed):</b> Eficiencia de tiempo.</li>
            <li><b>RC (Route Completion):</b> Porcentaje de la misión completado.</li>
            <li><b>TD (Total Distance):</b> Persistencia en la conducción.</li>
            <li><b>SR (Success Rate):</b> Logro del objetivo final.</li>
            <li><b>CS (Collision Speed):</b> Penalización por severidad de impacto.</li>
        </ul>
        <hr style="border-color: #334155;">
        <div class="row" style="font-size: 0.85rem;">
            <div class="col-md-6">
                <b style="color: #38bdf8;">🏆 Top UPI Score:</b> Promedio simple de UPI en los 15 escenarios.
            </div>
            <div class="col-md-6">
                <b style="color: #10b981;">🛡️ Top Consistency:</b> Calculado como <code>1.0 - Desviación Estándar</code>. Premia modelos con rendimiento uniforme (sin fallos catastróficos en mapas específicos).
            </div>
        </div>
    </div>

    <div class="grid">
        <div class="card stat-card blue">
            <h3>🏆 Top 5 (UPI Score)</h3>
            <ul id="best-avg"></ul>
        </div>
        <div class="card stat-card green">
            <h3>🛡️ Top 5 (Consistency)</h3>
            <ul id="best-stability"></ul>
        </div>
        <div class="card stat-card orange">
            <h3>🧐 Scenario Insights</h3>
            <div id="swot"></div>
        </div>
    </div>

    <div class="card">
        <h3>📍 Intelligence Heatmap (UPI)</h3>
        <p><i>Eje X: Checkpoints (Pasos) | Eje Y: Escenarios. El color indica la calidad integral de conducción.</i></p>
        <div id="chart"></div>
    </div>

    <script>
        const data = REPLACE_ME_DATA;

        const avgList = document.getElementById('best-avg');
        data.best_models.by_avg_success.slice(0, 5).forEach(m => {
            avgList.innerHTML += `<li><b>Step ${m.step}</b>: ${(m.avg_success * 100).toFixed(1)}%</li>`;
        });

        const stabList = document.getElementById('best-stability');
        data.best_models.by_stability.slice(0, 5).forEach(m => {
            stabList.innerHTML += `<li><b>Step ${m.step}</b>: Score ${(m.stability * 100).toFixed(1)}</li>`;
        });

        const series = data.scenarios.map(s => {
            return {
                name: s,
                data: data.steps.map(step => {
                    const stepRow = data.heatmap.find(r => r.step === step);
                    return {
                        x: step.toLocaleString(),
                        y: stepRow['eval' + s] || stepRow[s] || 0
                    };
                })
            };
        });

        new ApexCharts(document.querySelector("#chart"), {
            series: series,
            chart: { 
                height: 600, 
                type: 'heatmap',
                foreColor: '#94a3b8',
                toolbar: { show: true }
            },
            dataLabels: { enabled: false },
            plotOptions: {
                heatmap: {
                    shadeIntensity: 0.5,
                    colorScale: {
                        ranges: [
                            { from: 0, to: 0.4, name: 'Failed/Unsafe', color: '#ef4444' },
                            { from: 0.41, to: 0.7, name: 'Average', color: '#f59e0b' },
                            { from: 0.71, to: 0.85, name: 'High Performance', color: '#10b981' },
                            { from: 0.86, to: 1.0, name: 'Elite (Balanced)', color: '#38bdf8' }
                        ]
                    }
                }
            },
            xaxis: {
                title: { text: 'Training Steps' }
            }
        }).render();

        const scTotals = {};
        data.scenarios.forEach(s => scTotals[s] = 0);
        data.heatmap.forEach(r => {
            data.scenarios.forEach(s => scTotals[s] += (r['eval'+s] || r[s] || 0));
        });
        const scAvg = data.scenarios.map(s => ({ name: s, avg: scTotals[s]/data.steps.length }))
            .sort((a,b) => b.avg - a.avg);
        
        document.getElementById('swot').innerHTML = `
            <b>💪 Mejor Escenario:</b> ${scAvg[0].name} (${(scAvg[0].avg*100).toFixed(0)}% UPI)<br>
            <b>⚠️ Mayor Desafío:</b> ${scAvg[scAvg.length-1].name} (${(scAvg[scAvg.length-1].avg*100).toFixed(0)}% UPI)
        `;
    </script>
</body>
</html>
"""

final_html = html_template.replace("REPLACE_ME_DATA", json_data_str)

with open('advanced_analysis/heatmap.html', 'w') as f:
    f.write(final_html)

print("Dashboard generated successfully: advanced_analysis/heatmap.html")
