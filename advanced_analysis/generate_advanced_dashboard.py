import json
import os

def generate_advanced_dashboard():
    data_path = '/media/nemesis/disco4tb/Documents/VLM-RL/advanced_analysis/data/advanced_metrics.json'
    html_path = '/media/nemesis/disco4tb/Documents/VLM-RL/advanced_analysis/vlmrl_scenario_analysis.html'
    
    with open(data_path, 'r') as f:
        data = json.load(f)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>VLM-RL: Scenario & Stability Analysis</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <style>
            body {{ background-color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            .sidebar {{ background: #1e293b; color: white; padding: 20px; border-radius: 0 0 20px 0; }}
            .card {{ border-radius: 15px; border: none; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 25px; }}
            .header-banner {{ background: linear-gradient(135deg, #0f172a 0%, #334155 100%); color: white; padding: 40px 0; margin-bottom: 30px; border-bottom: 5px solid #3b82f6; }}
            .plot-container {{ height: 500px; }}
            .filter-box {{ background: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="header-banner text-center">
            <h1>🌍 VLM-RL: Análisis por Escenarios y Robustez</h1>
            <p class="lead">Comparativa de estabilidad (Jitter) en Towns 01-05 con tráfico Vacío vs Denso</p>
        </div>

        <div class="container-fluid">
            <div class="row px-4">
                <!-- Filtros -->
                <div class="col-md-3">
                    <div class="card p-4">
                        <h4 class="fw-bold mb-3">🔍 Filtros</h4>
                        <div class="mb-3">
                            <label class="form-label">Seleccionar Escenario (Town)</label>
                            <select id="town-select" class="form-select" onchange="updateCharts()">
                                <option value="All">Todos los Towns</option>
                                <option value="Town01">Town 01 (Simple)</option>
                                <option value="Town03">Town 03 (Complejo)</option>
                                <option value="Town04">Town 04 (Autovía)</option>
                                <option value="Town05">Town 05 (Urbano)</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Tipo de Tráfico</label>
                            <select id="traffic-select" class="form-select" onchange="updateCharts()">
                                <option value="All">Todos</option>
                                <option value="Empty">Vacío (Empty)</option>
                                <option value="Dense">Denso (Dense)</option>
                                <option value="Normal">Normal</option>
                            </select>
                        </div>
                        <hr>
                        <div class="alert alert-info small">
                            <strong>Jitter:</strong> Menor es mejor. Indica una conducción más suave y parecida a la humana.
                        </div>
                    </div>
                </div>

                <!-- Gráficos Principales -->
                <div class="col-md-9">
                    <div class="row">
                        <div class="col-12">
                            <div class="card p-4">
                                <h4 class="fw-bold">📊 Estabilidad vs Tiempo (Training Steps)</h4>
                                <div id="main-timeline" class="plot-container"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card p-4">
                                <h4 class="fw-bold">🏙️ Jitter por Town</h4>
                                <div id="town-box-plot" class="plot-container"></div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card p-4">
                                <h4 class="fw-bold">🚗 Impacto del Tráfico</h4>
                                <div id="traffic-impact-plot" class="plot-container"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const fullData = {json.dumps(data)};

            function updateCharts() {{
                const selectedTown = document.getElementById('town-select').value;
                const selectedTraffic = document.getElementById('traffic-select').value;

                let filtered = fullData;
                if (selectedTown !== 'All') filtered = filtered.filter(d => d.town === selectedTown);
                if (selectedTraffic !== 'All') filtered = filtered.filter(d => d.traffic === selectedTraffic);

                renderTimeline(filtered);
                renderTownBox(filtered);
                renderTrafficImpact(filtered);
            }}

            function renderTimeline(data) {{
                const sorted = [...data].sort((a, b) => a.steps - b.steps);
                
                // Agrupar por escenario para líneas múltiples
                const scenarios = [...new Set(sorted.map(d => d.folder))];
                const traces = scenarios.map(s => {{
                    const sData = sorted.filter(d => d.folder === s);
                    return {{
                        x: sData.map(d => d.steps),
                        y: sData.map(d => d.jitter),
                        name: s,
                        type: 'scatter',
                        mode: 'lines',
                        opacity: 0.6
                    }};
                }});

                Plotly.newPlot('main-timeline', traces, {{
                    xaxis: {{ title: 'Training Steps' }},
                    yaxis: {{ title: 'Steer Jitter (Lower is Better)', range: [0, 0.4] }},
                    legend: {{ orientation: 'h', y: -0.2 }},
                    margin: {{ t: 30 }}
                }});
            }}

            function renderTownBox(data) {{
                const trace = {{
                    y: data.map(d => d.jitter),
                    x: data.map(d => d.town),
                    type: 'box',
                    name: 'Jitter Dist',
                    marker: {{ color: '#3b82f6' }}
                }};
                Plotly.newPlot('town-box-plot', [trace], {{
                    yaxis: {{ title: 'Jitter' }},
                    xaxis: {{ title: 'Town' }},
                    margin: {{ t: 30 }}
                }});
            }}

            function renderTrafficImpact(data) {{
                const trafficTypes = [...new Set(data.map(d => d.traffic))];
                const traces = trafficTypes.map(t => {{
                    const tData = data.filter(d => d.traffic === t);
                    return {{
                        x: [t],
                        y: [tData.reduce((a, b) => a + b.jitter, 0) / tData.length],
                        type: 'bar',
                        name: t
                    }};
                }});
                Plotly.newPlot('traffic-impact-plot', traces, {{
                    yaxis: {{ title: 'Avg Jitter' }},
                    margin: {{ t: 30 }}
                }});
            }}

            // Inicializar
            updateCharts();
        </script>
    </body>
    </html>
    """
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"Dashboard generado: {{html_path}}")

if __name__ == "__main__":
    generate_advanced_dashboard()
