import json
import os
from collections import Counter

def generate_reports(json_file, suffix):
    json_path = os.path.join("data", json_file)
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        return
        
    with open(json_path, "r") as f:
        data_all = json.load(f)

    # Definiciones de Targets
    scenario_targets = {
        "Town01": {"as": 22.9, "rc": 1.00, "td": 5697.6, "cs": 0.03, "sr": 1.00},
        "Town02": {"as": 19.3, "rc": 0.97, "td": 2028.2, "cs": 0.02, "sr": 0.93},
        "Town03": {"as": 21.7, "rc": 0.91, "td": 3757.8, "cs": 1.14, "sr": 0.87},
        "Town04": {"as": 22.0, "rc": 0.80, "td": 12684.1, "cs": 2.15, "sr": 0.70},
        "Town05": {"as": 22.9, "rc": 0.93, "td": 3322.5, "cs": 0.46, "sr": 0.87},
        "emptyTown02": {"as": 23.8, "rc": 1.00, "td": 2113.9, "cs": 0.00, "sr": 1.00},
        "denseTown02": {"as": 16.1, "rc": 0.87, "td": 1819.0, "cs": 0.11, "sr": 0.80}
    }

    # --- 1. GENERAR DASHBOARD GLOBAL ---
    html_eval = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>VLM-RL: Global Dashboard {suffix}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
        body {{ background-color: #f8fafc; color: #1e293b; padding-bottom: 100px; font-family: 'Inter', sans-serif; }}
        .header-main {{ background: #1e293b; color: white; padding: 40px; margin-bottom: 20px; border-radius: 0 0 30px 30px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }}
        .control-panel {{ background: #334155; color: white; padding: 25px; border-radius: 20px; margin-bottom: 30px; border: 3px solid #fbbf24; }}
        .card {{ border-radius: 16px; margin-bottom: 30px; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.05); background: white; overflow: hidden; }}
        .plot-container {{ height: 450px; }}
        #zoomValue {{ color: #fbbf24; font-size: 1.5rem; font-weight: bold; }}
        .target-box {{ background: #1e293b; padding: 12px; border-radius: 12px; font-size: 0.9rem; margin-top: 15px; border: 1px solid #fbbf24; }}
    </style>
</head>
<body>
    <div class="header-main text-center">
        <h1 class="fw-bold display-4">🌍 VLM-RL Dashboard ({suffix})</h1>
        <p class="lead text-info">Análisis de Ejecución {suffix} - 15 Escenarios CARLA</p>
    </div>

    <div class="container-fluid px-5">
        <div class="row justify-content-center">
            <div class="col-md-6">
                <div class="control-panel shadow-lg">
                    <div class="row align-items-center">
                        <div class="col-md-6">
                            <label class="form-label fw-bold mb-2">🏙️ Seleccionar Escenario</label>
                            <select id="scenario-select" class="form-select form-select-lg bg-dark text-white border-info" onchange="updateDashboard()">
                                {"".join([f'<option value="{s}">{s}</option>' for s in sorted(data_all.keys())])}
                            </select>
                            <div id="target-display" class="target-box"></div>
                        </div>
                        <div class="col-md-6 text-center border-start border-secondary">
                            <label for="zoomSlider" class="form-label fw-bold">🔍 Zoom de Precisión: <span id="zoomValue">10</span>%</label>
                            <input type="range" class="form-range" min="1" max="100" value="10" id="zoomSlider" oninput="updateDashboard()">
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-12"><div class="card p-3"><div id="as-plot" class="plot-container"></div></div></div>
            <div class="col-12"><div class="card p-3"><div id="rc-plot" class="plot-container"></div></div></div>
            <div class="col-12"><div class="card p-3"><div id="td-plot" class="plot-container"></div></div></div>
            <div class="col-12"><div class="card p-3"><div id="sr-plot" class="plot-container"></div></div></div>
            <div class="col-12"><div class="card p-3"><div id="cs-plot" class="plot-container"></div></div></div>
        </div>
    </div>

    <script>
        const data_all = {json.dumps(data_all)};
        const scenarioTargets = {json.dumps(scenario_targets)};

        function updateDashboard() {{
            const scenario = document.getElementById('scenario-select').value;
            const zoomPercent = document.getElementById('zoomSlider').value / 100;
            document.getElementById('zoomValue').innerText = document.getElementById('zoomSlider').value;
            
            const target = scenarioTargets[scenario] || null;
            const data = data_all[scenario];
            const steps = data.map(d => d.steps);

            const targetBox = document.getElementById('target-display');
            if (target) {{
                targetBox.innerHTML = `<span style="color:#fbbf24">✔ Benchmark:</span> AS: ${{target.as}} | RC: ${{target.rc}} | TD: ${{target.td}} | CS: ${{target.cs}} | SR: ${{target.sr}}`;
            }} else {{
                targetBox.innerHTML = `<span style="color:#94a3b8">⚠ Sin benchmark oficial</span>`;
            }}

            const renderPlot = (id, key, title, color, targetVal, isInverse = false) => {{
                const vals = data.map(d => d[key]);
                const maxVal = Math.max(...vals, targetVal || 0);
                const minVal = Math.min(...vals, targetVal || 0);
                const delta = (maxVal - minVal) || 1;

                let yRange = isInverse ? [minVal - (delta * 0.05), minVal + (delta * zoomPercent)] : [maxVal - (delta * zoomPercent), maxVal + (delta * 0.05)];

                const traces = [{{ x: steps, y: vals, name: 'Local', type: 'scatter', mode: 'lines+markers', line: {{color: color, width: 3}} }}];
                if (targetVal !== null) traces.push({{ x: [steps[0], steps[steps.length-1]], y: [targetVal, targetVal], name: 'Paper', mode: 'lines', line: {{dash: 'dash', color: 'red'}} }});

                Plotly.newPlot(id, traces, {{ title: {{ text: `<b>${{title}}</b>` }}, yaxis: {{ range: yRange }}, margin: {{ t: 60, b: 60, l: 80, r: 40 }} }});
            }};

            renderPlot('as-plot', 'as', 'Average Speed (AS)', '#3498db', target ? target.as : null);
            renderPlot('rc-plot', 'rc', 'Route Completion (RC)', '#9b59b6', target ? target.rc : null);
            renderPlot('td-plot', 'td', 'Total Distance (TD)', '#f1c40f', target ? target.td : null);
            renderPlot('sr-plot', 'sr', 'Success Rate (SR)', '#2ecc71', target ? target.sr : null);
            renderPlot('cs-plot', 'cs', 'Collision Speed (CS)', '#e67e22', target ? target.cs : null, true);
        }}
        updateDashboard();
    </script>
</body>
</html>"""

    with open(f"analysis/vlmrl_global_evaluation_dashboard_{suffix}.html", "w") as f:
        f.write(html_eval)

    # --- 2. GENERAR ANÁLISIS DE ROBUSTEZ (CON MOSAICO COMPLETO) ---
    cat_groups = {
        "Global": list(data_all.keys()),
        "Empty": [s for s in data_all.keys() if "empty" in s],
        "Regular": [s for s in data_all.keys() if "empty" not in s and "dense" not in s],
        "Dense": [s for s in data_all.keys() if "dense" in s]
    }
    towns = ["Town01", "Town02", "Town03", "Town04", "Town05"]
    town_groups = {t: [s for s in data_all.keys() if t in s] for t in towns}
    all_groups = {**cat_groups, **town_groups}
    
    group_frequencies = {}
    scenario_tops = {}

    for group_name, scenarios in all_groups.items():
        all_top_checkpoints = []
        for scenario in scenarios:
            results = data_all.get(scenario, [])
            if not results: continue
            
            max_vals = {
                "as": max(d["as"] for d in results) or 1, "rc": max(d["rc"] for d in results) or 1,
                "td": max(d["td"] for d in results) or 1, "sr": max(d["sr"] for d in results) or 1,
                "cs": max(d["cs"] for d in results) or 1
            }
            
            for d in results:
                d["score"] = ((d["as"]/max_vals["as"]) + (d["rc"]/max_vals["rc"]) + 
                              (d["td"]/max_vals["td"]) + (d["sr"]/max_vals["sr"]) + 
                              (1-(d["cs"]/(max_vals["cs"] or 1)))) / 5 * 100

            top_10 = sorted(results, key=lambda x: x["score"], reverse=True)[:10]
            if group_name == "Global":
                scenario_tops[scenario] = top_10
            
            for m in top_10: all_top_checkpoints.append(m["steps"])
        
        freq = Counter(all_top_checkpoints)
        sf = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        group_frequencies[group_name] = {"labels": [str(x[0]) for x in sf], "values": [x[1] for x in sf]}

    html_robust = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>VLM-RL: Global Robustness {suffix}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ background-color: #0f172a; color: #e2e8f0; font-family: 'Inter', sans-serif; padding: 40px; }}
        .chart-container {{ background: #1e293b; border-radius: 20px; padding: 25px; margin-bottom: 40px; border: 1px solid #334155; }}
        .chart-main {{ border: 1px solid #fbbf24; }}
        .card-scen {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 15px; margin-bottom: 20px; font-size: 0.8rem; }}
        .table-mini {{ color: #94a3b8; width: 100%; }}
        .table-mini th {{ color: #f8fafc; font-size: 0.7rem; border-bottom: 1px solid #334155; }}
        .highlight-row {{ background: rgba(251, 191, 36, 0.1); color: #fbbf24; font-weight: bold; }}
        .category-title {{ color: #fbbf24; border-left: 4px solid #fbbf24; padding-left: 15px; margin: 50px 0 25px 0; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="text-center mb-5">
        <h1 class="display-4 fw-bold">🏆 Robustness Championship ({suffix})</h1>
        <p class="lead text-warning">Frecuencia de Élite por Categoría y Ciudad</p>
    </div>

    <div class="chart-container chart-main shadow-lg">
        <h3 class="text-center mb-4">⭐ DOMINANCIA GLOBAL</h3>
        <div id="chart-Global" style="height: 400px;"></div>
    </div>

    <div class="row">
        <div class="col-lg-4"><div class="chart-container shadow-sm"><h4 class="text-center text-info">🍃 Empty</h4><div id="chart-Empty" style="height: 250px;"></div></div></div>
        <div class="col-lg-4"><div class="chart-container shadow-sm"><h4 class="text-center text-success">🚗 Regular</h4><div id="chart-Regular" style="height: 250px;"></div></div></div>
        <div class="col-lg-4"><div class="chart-container shadow-sm"><h4 class="text-center text-danger">🛑 Dense</h4><div id="chart-Dense" style="height: 250px;"></div></div></div>
    </div>

    <h2 class="category-title">🏙️ Especialistas por Ciudad</h2>
    <div class="row">
        {"".join([f'<div class="col" style="width:20%"><div class="chart-container" style="padding:10px;"><h6 class="text-center text-warning">{t}</h6><div id="chart-{t}" style="height:180px;"></div></div></div>' for t in towns])}
    </div>

    <h2 class="category-title">📂 Desglose por Escenario Individual</h2>
    <div class="row">
        {"".join([f'''
        <div class="col-xl-3 col-lg-4 col-md-6">
            <div class="card-scen shadow-sm">
                <div class="fw-bold text-info mb-2">{s}</div>
                <table class="table-mini">
                    <thead><tr><th>#</th><th>Steps</th><th>Score</th></tr></thead>
                    <tbody>
                        {"".join([f'<tr class="{ "highlight-row" if i<3 else "" }"><td>{i+1}</td><td>{m["steps"]}</td><td>{m["score"]:.1f}%</td></tr>' for i, m in enumerate(tops)])}
                    </tbody>
                </table>
            </div>
        </div>
        ''' for s, tops in sorted(scenario_tops.items())])}
    </div>

    <script>
        const freqs = {json.dumps(group_frequencies)};
        function plot(divId, data, color, size=10) {{
            const trace = {{ x: data.labels, y: data.values, type: 'bar', marker: {{ color: color }} }};
            Plotly.newPlot(divId, [trace], {{
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: '#f8fafc', size: size }}, margin: {{ t: 10, b: 60, l: 30, r: 5 }},
                xaxis: {{ tickangle: -45 }}, yaxis: {{ gridcolor: '#334155', dtick: 1 }}
            }});
        }}
        plot('chart-Global', freqs.Global, '#fbbf24', 12);
        plot('chart-Empty', freqs.Empty, '#3498db');
        plot('chart-Regular', freqs.Regular, '#2ecc71');
        plot('chart-Dense', freqs.Dense, '#e74c3c');
        { "".join([f"plot('chart-{t}', freqs['{t}'], '#94a3b8', 8);" for t in towns]) }
    </script>
</body>
</html>"""

    with open(f"analysis/vlmrl_global_robustness_analysis_{suffix}.html", "w") as f:
        f.write(html_robust)
    print(f"Generated FULL analysis for {suffix}")

if __name__ == "__main__":
    import sys
    generate_reports(sys.argv[1], sys.argv[2])
