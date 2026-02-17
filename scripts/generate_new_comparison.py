import os
import json
from tensorboard.backend.event_processing import event_accumulator

def extract_tb_data(path, label):
    print(f"Extracting data for {label}...")
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    # Mapeo de tags reales a nombres amigables
    name_map = {
        'custom/mean_reward': 'Mean Reward',
        'custom/routes_completed': 'Routes Completed',
        'custom/avg_speed': 'Average Speed',
        'custom/collision_rate': 'Collision Rate',
        'custom/mean_steer_smoothness_x100': 'Steer Smoothness (x100)',
        'rollout/ep_gt_rew_mean': 'GT Reward Mean'
    }
    
    data = {}
    for tag, friendly_name in name_map.items():
        if tag in tags:
            data[friendly_name] = [{'step': e.step, 'value': e.value} for e in ea.Scalars(tag)]
            
    return data

def generate_comparison_dashboard():
    path_baseline = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0"
    path_ours = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"
    
    baseline_data = extract_tb_data(path_baseline, "VLM-RL (baseline)")
    ours_data = extract_tb_data(path_ours, "CLG-Smooth (ours)")
    
    combined_data = {
        "VLM-RL (baseline)": baseline_data,
        "CLG-Smooth (ours)": ours_data
    }
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>VLM-RL vs CLG-Smooth: Comparison Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ background-color: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 30px; }}
            .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 15px; margin-bottom: 30px; padding: 20px; }}
            h1 {{ color: #38bdf8; font-weight: 800; text-transform: uppercase; letter-spacing: 2px; text-align: center; margin-bottom: 40px; }}
            .metric-title {{ color: #f59e0b; font-weight: bold; margin-bottom: 20px; border-left: 5px solid #38bdf8; padding-left: 15px; }}
            .plot-container {{ height: 450px; }}
            .header-info {{ background: #0c4a6e; padding: 20px; border-radius: 15px; margin-bottom: 30px; border: 1px solid #38bdf8; }}
        </style>
    </head>
    <body>
        <div class="header-info text-center">
            <h1>🏆 VLM-RL (Baseline) vs CLG-Smooth (Ours)</h1>
            <p class="lead">Comparativa de entrenamiento: Estabilidad y Rendimiento</p>
        </div>
        
        <div class="container-fluid">
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <h3 class="metric-title">📈 Routes Completed</h3>
                        <div id="plot-routes" class="plot-container"></div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <h3 class="metric-title">💰 Mean Reward</h3>
                        <div id="plot-reward" class="plot-container"></div>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <h3 class="metric-title">✨ Steer Smoothness</h3>
                        <div id="plot-smooth" class="plot-container"></div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <h3 class="metric-title">⚡ Average Speed</h3>
                        <div id="plot-speed" class="plot-container"></div>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <h3 class="metric-title">⚠️ Collision Rate</h3>
                        <div id="plot-collision" class="plot-container"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const data = {json.dumps(combined_data)};
            
            function createPlot(id, metricName) {{
                const traces = [];
                const colors = {{
                    "VLM-RL (baseline)": "#94a3b8", // Gris
                    "CLG-Smooth (ours)": "#38bdf8"   // Celeste brillante
                }};

                for (const runLabel in data) {{
                    const runData = data[runLabel][metricName];
                    if (runData) {{
                        traces.push({{
                            x: runData.map(d => d.step),
                            y: runData.map(d => d.value),
                            name: runLabel,
                            type: 'scatter',
                            mode: 'lines',
                            line: {{ color: colors[runLabel], width: 3 }}
                        }});
                    }}
                }}

                if (traces.length === 0) return;

                Plotly.newPlot(id, traces, {{
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {{ color: '#94a3b8' }},
                    xaxis: {{ title: 'Training Steps', gridcolor: '#334155' }},
                    yaxis: {{ title: metricName, gridcolor: '#334155' }},
                    legend: {{ orientation: 'h', y: 1.1 }},
                    margin: {{ t: 40, b: 60, l: 60, r: 20 }}
                }});
            }}

            createPlot('plot-routes', 'Routes Completed');
            createPlot('plot-reward', 'Mean Reward');
            createPlot('plot-smooth', 'Steer Smoothness (x100)');
            createPlot('plot-speed', 'Average Speed');
            createPlot('plot-collision', 'Collision Rate');
        </script>
    </body>
    </html>
    """
    
    with open("tensorboard_analysis/comparison_dashboard_nuevo.html", "w") as f:
        f.write(html_content)
    print("✅ Dashboard 'comparison_dashboard_nuevo.html' generado exitosamente con los tags correctos.")

if __name__ == "__main__":
    generate_comparison_dashboard()
