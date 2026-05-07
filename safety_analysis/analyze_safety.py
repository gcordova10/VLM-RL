import os
import pandas as pd
import json
import glob
import re
import sys

def clean_val(val):
    if pd.isna(val): return 0.0
    s = str(val).strip()
    if not s: return 0.0
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    if matches:
        return float(matches[-1])
    return 0.0

def generate_safety_report(base_path, output_path):
    # Definición de mapeo detallado
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
        full_path = os.path.join(base_path, folder)
        if not os.path.exists(full_path): continue
            
        csv_files = glob.glob(os.path.join(full_path, "*_eval_summary.csv"))
        for f_path in csv_files:
            try:
                match = re.search(r"model_(\d+)_steps", os.path.basename(f_path))
                if not match: continue
                steps = int(match.group(1))
                df = pd.read_csv(f_path, dtype=object)
                total_row = df[df['episode'].astype(str).str.lower() == 'total']
                
                if not total_row.empty:
                    row = total_row.iloc[0]
                    sr = clean_val(row.get('success', 0))
                    cs = clean_val(row.get('collision_speed', 0))
                    cd_mean = clean_val(row.get('center_dev_mean', 0))
                    cd_std = clean_val(row.get('center_dev_std', 0))
                    speed = clean_val(row.get('speed_mean', 0))
                    
                    # Score de Seguridad por escenario
                    coll_pen = 50 if cs > 0 else 0
                    prec_pen = min(30, cd_mean * 20)
                    stab_pen = min(20, cd_std * 20)
                    safety_score = max(0, (sr * 100) - coll_pen - prec_pen - stab_pen)
                    
                    all_results.append({
                        "steps": steps,
                        "town": town,
                        "density": density,
                        "sr": sr,
                        "cs": cs,
                        "cd_mean": cd_mean,
                        "cd_std": cd_std,
                        "speed": speed,
                        "safety_score": round(safety_score, 2)
                    })
            except: continue

    if not all_results:
        print("No data found.")
        return

    # Guardar JSON completo para el Dashboard interactivo
    json_data = json.dumps(all_results)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VLM-RL: Multi-Scenario Safety Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ background-color: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', Tahoma, sans-serif; padding: 30px; }}
            .control-panel {{ background: #1e293b; border: 1px solid #38bdf8; border-radius: 20px; padding: 25px; margin-bottom: 30px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3); }}
            .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }}
            .table {{ color: #e2e8f0; font-size: 0.9rem; }}
            .table-dark {{ --bs-table-bg: #1e293b; }}
            .highlight-gold {{ color: #f59e0b; font-weight: bold; }}
            select.form-select {{ background-color: #0f172a; color: #e2e8f0; border-color: #334155; }}
            .text-warning {{ color: #f59e0b !important; }}
            .text-info {{ color: #38bdf8 !important; }}
            .text-secondary {{ color: #94a3b8 !important; }}
            .bg-dark-soft {{ background: #0f172a; border: 1px solid #334155; }}
            h1 {{ color: #38bdf8; font-weight: 800; text-transform: uppercase; letter-spacing: 2px; }}
        </style>
    </head>
    <body>
        <h1 class="display-5 text-center mb-4">🛡️ Granular Safety Analysis</h1>

        <!-- BLOQUE EXPLICATIVO DE LA MÉTRICA -->
        <div class="card p-4 mb-4">
            <div class="row align-items-center">
                <div class="col-md-4 border-end border-secondary">
                    <h4 class="text-warning">🧮 Lógica de Evaluación</h4>
                    <p class="small text-secondary">Inspirado en la <b>"Hierarchical Reward Synthesis"</b> del paper VLM-RL, esta métrica audita la calidad técnica de la conducción post-entrenamiento.</p>
                    <div class="bg-dark-soft p-3 rounded text-center mb-3">
                        <code class="text-info h5">Safety Score = max(0, (SR × 100) - P_coll - P_prec - P_stab)</code>
                    </div>
                </div>
                <div class="col-md-8 ps-4 text-start">
                    <h5 class="text-warning">Desglose de Componentes:</h5>
                    <div class="row small">
                        <div class="col-md-6">
                            <p><b class="text-info">1. Punto de Partida (Base):</b> <code class="text-white">(SR × 100)</code><br>
                            <span class="text-light">Éxito binario (100 pts si llega, 0 pts si falla).</span></p>
                            
                            <p><b class="text-info">2. Penalización por Colisión (P_coll):</b> <code class="text-white">-50 pts</code><br>
                            <span class="text-light">Si <code>collision_speed > 0</code>. La seguridad es crítica.</span></p>
                        </div>
                        <div class="col-md-6 border-start border-secondary ps-3">
                            <p><b class="text-info">3. Precisión de Carril (P_prec):</b> <code class="text-white">Hasta -30 pts</code><br>
                            <span class="text-light">Penaliza la <b>Desviación Media (cd_mean)</b>. Mide qué tan lejos del centro del carril conduce el modelo.</span></p>
                            
                            <p><b class="text-info">4. Estabilidad / Zigzagueo (P_stab):</b> <code class="text-white">Hasta -20 pts</code><br>
                            <span class="text-light">Penaliza la <b>Variación (cd_std)</b>. Mide el "nerviosismo" o zigzagueo del volante mientras conduce.</span></p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- EXPLICACIÓN DE CUADRANTES PARETO -->
        <div class="card p-4 mb-4" style="border-left: 5px solid #38bdf8;">
            <h5 class="text-info mb-3">🎯 Interpretación del Gráfico de Pareto (Velocidad vs Seguridad)</h5>
            <div class="row text-center g-3">
                <div class="col-md-3">
                    <div class="p-3 rounded" style="background: #0c4a6e; border: 1px dashed #38bdf8;">
                        <div class="fw-bold text-info">🐢 El Prudente</div>
                        <div class="x-small text-secondary">Arriba-Izquierda. Seguro pero lento.</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="p-3 rounded" style="background: #064e3b; border: 2px solid #10b981;">
                        <div class="fw-bold text-success">🏆 El Experto (Elite)</div>
                        <div class="x-small text-secondary">Arriba-Derecha. El equilibrio perfecto.</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="p-3 rounded" style="background: #450a0a; border: 1px dashed #ef4444;">
                        <div class="fw-bold text-danger">⚠️ El Ineficiente</div>
                        <div class="x-small text-secondary">Abajo-Izquierda. Lento y peligroso.</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="p-3 rounded" style="background: #451a03; border: 1px solid #f59e0b;">
                        <div class="fw-bold text-warning">🏎️ El Temerario</div>
                        <div class="x-small text-secondary">Abajo-Derecha. Rápido pero arriesgado.</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="control-panel shadow-sm">
            <div class="row">
                <div class="col-md-4">
                    <label class="form-label fw-bold">🏙️ Seleccionar Ciudad</label>
                    <select id="town-filter" class="form-select" onchange="updateDashboard()">
                        <option value="all">Todas las ciudades (Promedio)</option>
                        <option value="Town01">Town01 (Base)</option>
                        <option value="Town02">Town02 (Urbano)</option>
                        <option value="Town03">Town03 (Complejo)</option>
                        <option value="Town04">Town04 (Largo/Autopista)</option>
                        <option value="Town05">Town05 (Densidad Calles)</option>
                    </select>
                </div>
                <div class="col-md-4">
                    <label class="form-label fw-bold">🚗 Densidad de Tráfico</label>
                    <select id="density-filter" class="form-select" onchange="updateDashboard()">
                        <option value="all">Todas las densidades</option>
                        <option value="empty">Empty (Navegación pura)</option>
                        <option value="regular">Regular (Tráfico normal)</option>
                        <option value="dense">Dense (Tráfico pesado)</option>
                    </select>
                </div>
                <div class="col-md-4 text-center d-flex align-items-center justify-content-center">
                    <div class="h4">Modelos Analizados: <span class="text-warning">100</span></div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-8">
                <div class="card p-4">
                    <h4 id="plot-title" class="text-info fw-bold">Velocidad vs Seguridad</h4>
                    <div id="pareto-plot" style="height: 500px;"></div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card p-4">
                    <h4 class="text-info fw-bold">🏆 Top 10 Elite</h4>
                    <div id="top-list" class="list-group list-group-flush bg-transparent">
                        <!-- Se llena con JS -->
                    </div>
                </div>
            </div>
        </div>

        <div class="card p-4 mt-4">
            <h4 class="text-info fw-bold mb-4">📂 Tabla de Ranking Detallada</h4>
            <div class="table-responsive">
                <table class="table table-dark table-hover small">
                    <thead>
                        <tr>
                            <th>Steps</th>
                            <th>Safety Score ↓</th>
                            <th>Success Rate</th>
                            <th>Avg Speed</th>
                            <th>Precision (m)</th>
                            <th>Collisions</th>
                        </tr>
                    </thead>
                    <tbody id="table-body">
                        <!-- Se llena con JS -->
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            const rawData = {json_data};

            function updateDashboard() {{
                const town = document.getElementById('town-filter').value;
                const density = document.getElementById('density-filter').value;
                
                let filtered = rawData;
                if (town !== 'all') filtered = filtered.filter(d => d.town === town);
                if (density !== 'all') filtered = filtered.filter(d => d.density === density);

                // Agrupar por steps
                const grouped = {{}};
                filtered.forEach(d => {{
                    if (!grouped[d.steps]) {{
                        grouped[d.steps] = {{ steps: d.steps, ss: [], sr: [], sp: [], cd: [], cs: 0 }};
                    }}
                    grouped[d.steps].ss.push(d.safety_score);
                    grouped[d.steps].sr.push(d.sr);
                    grouped[d.steps].sp.push(d.speed);
                    grouped[d.steps].cd.push(d.cd_mean);
                    grouped[d.steps].cs += d.cs;
                }});

                const stats = Object.values(grouped).map(g => ({{
                    steps: g.steps,
                    safety_score: g.ss.reduce((a,b) => a+b, 0) / g.ss.length,
                    sr: g.sr.reduce((a,b) => a+b, 0) / g.sr.length,
                    speed: g.sp.reduce((a,b) => a+b, 0) / g.sp.length,
                    cd_mean: g.cd.reduce((a,b) => a+b, 0) / g.cd.length,
                    cs: g.cs
                }})).sort((a,b) => b.safety_score - a.safety_score);

                renderUI(stats, town, density);
            }}

            function renderUI(data, town, density) {{
                document.getElementById('plot-title').innerText = `Velocidad vs Seguridad (${{town}} - ${{density}})`;
                
                // Plot
                const trace = {{
                    x: data.map(d => d.speed),
                    y: data.map(d => d.safety_score),
                    mode: 'markers',
                    type: 'scatter',
                    text: data.map(d => d.steps.toLocaleString() + ' steps'),
                    marker: {{ 
                        size: 12, 
                        color: data.map(d => d.safety_score), 
                        colorscale: 'Viridis', 
                        showscale: true,
                        line: {{ color: '#334155', width: 1 }}
                    }}
                }};
                Plotly.newPlot('pareto-plot', [trace], {{
                    paper_bgcolor: 'rgba(0,0,0,0)', 
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {{ color: '#94a3b8' }},
                    xaxis: {{ title: 'Velocidad Media', gridcolor: '#334155', zerolinecolor: '#334155' }},
                    yaxis: {{ title: 'Safety Score', gridcolor: '#334155', zerolinecolor: '#334155' }},
                    margin: {{ t: 10 }}
                }});

                // Top List
                const topList = document.getElementById('top-list');
                topList.innerHTML = data.slice(0, 10).map(d => `
                    <div class="list-group-item bg-transparent text-light border-secondary px-0">
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="highlight-gold">${{d.steps.toLocaleString()}}</span>
                            <span class="badge bg-dark border border-success text-success">${{d.safety_score.toFixed(1)}}</span>
                        </div>
                    </div>
                `).join('');

                // Table
                const tbody = document.getElementById('table-body');
                tbody.innerHTML = data.map(d => `
                    <tr>
                        <td class="fw-bold text-white">${{d.steps.toLocaleString()}}</td>
                        <td class="highlight-gold">${{d.safety_score.toFixed(1)}}</td>
                        <td class="text-secondary">${{(d.sr*100).toFixed(1)}}%</td>
                        <td class="text-secondary">${{d.speed.toFixed(2)}}</td>
                        <td class="text-secondary">${{d.cd_mean.toFixed(3)}}</td>
                        <td class="${{d.cs > 0 ? 'text-danger fw-bold' : 'text-secondary'}}">${{d.cs.toFixed(1)}}</td>
                    </tr>
                `).join('');
            }}

            updateDashboard();
        </script>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"✅ Dashboard generado: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 safety_analysis/analyze_safety.py <BASE_PATH> <OUTPUT_FILENAME>")
        sys.exit(1)
        
    base_path = sys.argv[1]
    output_filename = sys.argv[2]
    
    if not output_filename.endswith('.html'):
        output_filename += '.html'
        
    # Asegurar que la ruta de salida esté en safety_analysis/ si no se especifica otra cosa
    if not os.path.dirname(output_filename):
        output_filename = os.path.join('safety_analysis', output_filename)
        
    generate_safety_report(base_path, output_filename)