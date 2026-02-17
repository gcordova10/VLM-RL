import os
import pandas as pd
import glob
import json
import re

def calculate_all_metrics():
    base_path = '/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl'
    output_dir = '/media/nemesis/disco4tb/Documents/VLM-RL/driving_quality_analysis'
    
    scenarios_map = {
        "eval": ("Town02", "regular"),
        "evalempty": ("Town02", "empty"),
        "evaldense": ("Town02", "dense"),
        "evalTown01": ("Town01", "regular"),
        "evalemptyTown01": ("Town01", "empty"),
        "evaldenseTown01": ("Town01", "dense"),
        "evalTown03": ("Town03", "regular"),
        "evalemptyTown03": ("Town03", "empty"),
        "evaldenseTown03": ("Town03", "dense"),
        "evalTown04": ("Town04", "regular"),
        "evalemptyTown04": ("Town04", "empty"),
        "evaldenseTown04": ("Town04", "dense"),
        "evalTown05": ("Town05", "regular"),
        "evalemptyTown05": ("Town05", "empty"),
        "evaldenseTown05": ("Town05", "dense"),
    }
    
    all_results = []

    for folder, (town, density) in scenarios_map.items():
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path): continue
            
        csv_files = glob.glob(os.path.join(folder_path, "model_*_steps_eval.csv"))
        for f in csv_files:
            try:
                match = re.search(r"model_(\d+)_steps", os.path.basename(f))
                if not match: continue
                steps = int(match.group(1))
                df = pd.read_csv(f, on_bad_lines='skip')
                df_clean = df[pd.to_numeric(df['steer'], errors='coerce').notnull()].copy()
                df_clean['steer'] = df_clean['steer'].astype(float)
                df_clean['speed'] = df_clean['speed'].astype(float)
                
                if df_clean.empty: continue
                
                jitter = df_clean.groupby('episode')['steer'].std().mean()
                avg_speed = df_clean['speed'].mean()
                stability = max(0, 1 - (jitter * 2.5)) 
                
                all_results.append({
                    "steps": steps, "town": town, "density": density,
                    "jitter": round(float(jitter), 4), "speed": round(float(avg_speed), 2),
                    "stability": round(float(stability), 4)
                })
            except: continue

    # GENERAR HTML CON DATOS INYECTADOS
    json_data = json.dumps(all_results)
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>VLM-RL: Driving Quality Analysis</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ background-color: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 30px; }}
            .control-panel {{ background: #1e293b; border: 1px solid #38bdf8; border-radius: 20px; padding: 25px; margin-bottom: 30px; }}
            .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 15px; margin-bottom: 20px; }}
            .text-info {{ color: #38bdf8 !important; }}
            .text-warning {{ color: #f59e0b !important; }}
            .text-secondary {{ color: #94a3b8 !important; }}
            select.form-select {{ background-color: #0f172a; color: #e2e8f0; border-color: #334155; }}
            h1 {{ color: #38bdf8; font-weight: 800; text-transform: uppercase; letter-spacing: 2px; }}
        </style>
    </head>
    <body>
        <h1 class="text-center mb-4">🚗 Driving Quality & Control Analysis</h1>

        <div class="card p-4 mb-4">
            <h4 class="text-warning">🧠 Metodología: Detección de "Jitter"</h4>
            <p class="small text-secondary">Este análisis audita la estabilidad del control. Un modelo con Jitter alto conduce como un "borracho", dando volantazos constantes que lo hacen frágil ante el lag.</p>
            <div class="row small">
                <div class="col-md-6 border-end border-secondary">
                    <span class="text-info fw-bold">Steer Jitter: Desviación Estándar del <code>steer</code>. Mide la oscilación del volante.</span>
                </div>
                <div class="col-md-6 ps-3">
                    <span class="text-info fw-bold">Stability Score: <code>max(0, 1 - (Jitter × 2.5))</code>. 1.0 = Línea de dirección perfecta.</span>
                </div>
            </div>
        </div>

        <div class="control-panel">
            <div class="row">
                <div class="col-md-4">
                    <label class="form-label fw-bold">🏙️ Ciudad</label>
                    <select id="town-filter" class="form-select" onchange="updateDashboard()">
                        <option value="all">Todas (Promedio)</option>
                        <option value="Town01">Town01</option><option value="Town02">Town02</option>
                        <option value="Town03">Town03</option><option value="Town04">Town04</option>
                        <option value="Town05">Town05</option>
                    </select>
                </div>
                <div class="col-md-4">
                    <label class="form-label fw-bold">🚗 Densidad</label>
                    <select id="density-filter" class="form-select" onchange="updateDashboard()">
                        <option value="all">Todas</option>
                        <option value="empty">Empty</option><option value="regular">Regular</option>
                        <option value="dense">Dense</option>
                    </select>
                </div>
                <div class="col-md-4 text-center d-flex align-items-center justify-content-center">
                    <div class="h4">Checkpoints: <span class="text-warning">{len(all_results)//15 if all_results else 0}</span></div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-8">
                <div class="card p-4">
                    <h4 class="text-info">📈 Evolución de Estabilidad (Training Steps)</h4>
                    <div id="jitter-plot" style="height: 500px;"></div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card p-4">
                    <h4 class="text-warning">🏆 Top 10 Suavidad</h4>
                    <div id="top-list" class="list-group list-group-flush bg-transparent"></div>
                </div>
            </div>
        </div>

        <div class="card p-4">
            <h4 class="text-info">🎯 Velocidad vs Estabilidad (Pareto de Control)</h4>
            <div id="scatter-plot" style="height: 500px;"></div>
        </div>

        <script>
            const rawData = {json_data};

            function updateDashboard() {{
                const town = document.getElementById('town-filter').value;
                const density = document.getElementById('density-filter').value;
                
                let filtered = rawData;
                if (town !== 'all') filtered = filtered.filter(d => d.town === town);
                if (density !== 'all') filtered = filtered.filter(d => d.density === density);

                const grouped = {{}};
                filtered.forEach(d => {{
                    if (!grouped[d.steps]) grouped[d.steps] = {{ steps: d.steps, jit: [], sp: [], stab: [] }};
                    grouped[d.steps].jit.push(d.jitter);
                    grouped[d.steps].sp.push(d.speed);
                    grouped[d.steps].stab.push(d.stability);
                }});

                const stats = Object.values(grouped).map(g => ({{
                    steps: g.steps,
                    jitter: g.jit.reduce((a,b) => a+b, 0) / g.jit.length,
                    speed: g.sp.reduce((a,b) => a+b, 0) / g.sp.length,
                    stability: g.stab.reduce((a,b) => a+b, 0) / g.stab.length
                }})).sort((a,b) => a.steps - b.steps);

                renderPlots(stats);
                renderTopList(stats);
            }}

            function renderPlots(data) {{
                Plotly.newPlot('jitter-plot', [{{
                    x: data.map(d => d.steps), y: data.map(d => d.jitter),
                    type: 'scatter', mode: 'lines+markers', line: {{ color: '#ef4444', width: 2 }}
                }}], {{
                    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {{ color: '#94a3b8' }},
                    xaxis: {{ title: 'Steps', gridcolor: '#334155' }},
                    yaxis: {{ title: 'Jitter', gridcolor: '#334155' }}
                }});

                Plotly.newPlot('scatter-plot', [{{
                    x: data.map(d => d.speed), y: data.map(d => d.stability),
                    mode: 'markers', text: data.map(d => d.steps.toLocaleString()),
                    marker: {{ size: 10, color: data.map(d => d.stability), colorscale: 'Portland', showscale: true }}
                }}], {{
                    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {{ color: '#94a3b8' }},
                    xaxis: {{ title: 'Speed (km/h)', gridcolor: '#334155' }},
                    yaxis: {{ title: 'Stability Score', gridcolor: '#334155' }}
                }});
            }}

            function renderTopList(data) {{
                const sorted = [...data].sort((a,b) => b.stability - a.stability).slice(0, 10);
                document.getElementById('top-list').innerHTML = sorted.map(d => `
                    <div class="list-group-item bg-transparent text-light border-secondary px-0 d-flex justify-content-between">
                        <span class="text-info">${{d.steps.toLocaleString()}}</span>
                        <span class="badge bg-dark border border-warning text-warning">${{(d.stability * 100).toFixed(1)}}% Stab</span>
                    </div>
                `).join('');
            }}

            updateDashboard();
        </script>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "driving_quality_dashboard.html"), "w") as f:
        f.write(html_template)
    
    print(f"✅ Dashboard generado con {len(all_results)} registros.")

if __name__ == "__main__":
    calculate_all_metrics()
