import os
import pandas as pd
import glob
import json

def generate_full_evaluation_dashboard():
    # Referencia oficial del Paper (VLM-RL)
    target = {'AS': 19.3, 'RC': 0.97, 'TD': 2028.2, 'CS': 0.02, 'SR': 0.93}
    
    eval_dir = '/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/eval'
    files = sorted(glob.glob(os.path.join(eval_dir, 'model_*_steps_eval_summary.csv')))
    
    data = []
    elite_data = []
    
    # Primero recolectamos máximos para normalizar
    max_as = 0
    max_td = 0
    
    for f in files:
        try:
            steps = int(os.path.basename(f).split('_')[1])
            df = pd.read_csv(f)
            row = df[df['episode'] == 'total']
            if row.empty: continue
            
            as_v = float(row['speed_mean'].values[0])
            rc_v = float(row['routes_completed'].values[0])
            td_v = float(row['total_distance'].values[0])
            cs_v = float(row['collision_speed'].values[0])
            sr_raw = row['success'].values[0]
            sr_v = 1.0 if str(sr_raw).lower() in ['true', '1', '1.0'] else (0.0 if str(sr_raw).lower() in ['false', '0', '0.0'] else float(sr_raw))
            
            if as_v > max_as: max_as = as_v
            if td_v > max_td: max_td = td_v

            item = {
                'steps': steps, 'as': round(as_v, 2), 'rc': round(rc_v, 2),
                'td': round(td_v, 2), 'cs': round(cs_v, 4), 'sr': round(sr_v, 2)
            }
            data.append(item)
            
            # FILTRO ELITE
            if (as_v > 19.3 and rc_v > 0.97 and td_v > 2028 and cs_v < 0.02 and sr_v > 0.93):
                elite_data.append(item)
                
        except: continue

    # Función para normalizar: Paper Target = 0.8, Máximo Local = 1.0
    def normalize(val, target_val, max_val):
        if val <= target_val:
            # Si es menor al target, mapeamos de 0 a 0.8
            return (val / target_val) * 0.8
        else:
            # Si supera al target, mapeamos de 0.8 a 1.0
            return 0.8 + 0.2 * (val - target_val) / (max_val - target_val)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>VLM-RL: AS vs TD Elite Analysis</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <style>
            body {{ background-color: #f8fafc; padding-bottom: 100px; font-family: 'Segoe UI', Tahoma, sans-serif; }}
            .card {{ border-radius: 15px; border: none; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); margin-bottom: 30px; }}
            .elite-header {{ background: #0f172a; color: white; padding: 40px 0; margin-bottom: 40px; border-bottom: 6px solid #f59e0b; }}
            .plot-container {{ height: 600px; }}
        </style>
    </head>
    <body>
        <div class="elite-header text-center">
            <h1>🏆 Comparativa Elite: Velocidad vs Distancia</h1>
            <p class="lead">Análisis enfocado en los {len(elite_data)} modelos que superaron al Paper</p>
        </div>

        <div class="container">
            <div class="row">
                <div class="col-12">
                    <div class="card p-4">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h3 class="fw-bold">🚀 AS y TD Normalizados (Zoom 0.94 - 1.0)</h3>
                            <span class="badge bg-danger">Línea en 0.8 = Desempeño del Paper</span>
                        </div>
                        <div id="elite-bar-plot" class="plot-container"></div>
                    </div>
                </div>
            </div>

            <div class="row mt-2">
                <div class="col-md-12">
                    <div class="card p-4 bg-dark text-white">
                        <h4>🔍 Resumen de Datos Brutos (Elite)</h4>
                        <table class="table table-dark table-hover small">
                            <thead>
                                <tr><th>Checkpoint</th><th>AS (km/h)</th><th>TD (metros)</th><th>Éxito</th><th>CS</th></tr>
                            </thead>
                            <tbody>
                                {"".join([f"<tr><td>{r['steps']:,}</td><td>{r['as']}</td><td>{r['td']}</td><td>{int(r['sr']*100)}%</td><td>{r['cs']}</td></tr>" for r in sorted(elite_data, key=lambda x: x['as'], reverse=True)])}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const elite_data = {json.dumps(elite_data)};
            const target = {json.dumps(target)};
            const max_as = {max_as};
            const max_td = {max_td};

            const normalize = (val, targetVal, maxVal) => {{
                if (val <= targetVal) return (val / targetVal) * 0.8;
                return 0.8 + 0.2 * (val - targetVal) / (maxVal - targetVal);
            }};

            if (elite_data.length > 0) {{
                const steps_labels = elite_data.map(r => r.steps.toLocaleString());
                
                const traceAS = {{
                    x: steps_labels,
                    y: elite_data.map(r => normalize(r.as, target.AS, max_as)),
                    name: 'Velocidad (AS)',
                    type: 'bar',
                    marker: {{color: 'blue'}}
                }};

                const traceTD = {{
                    x: steps_labels,
                    y: elite_data.map(r => normalize(r.td, target.TD, max_td)),
                    name: 'Distancia (TD)',
                    type: 'bar',
                    marker: {{color: 'red'}}
                }};

                Plotly.newPlot('elite-bar-plot', [traceAS, traceTD], {{
                    barmode: 'group',
                    yaxis: {{
                        title: 'Desempeño Relativo (Paper = 0.8)',
                        range: [0.94, 1.01],
                        gridcolor: '#e2e8f0'
                    }},
                    xaxis: {{ title: 'Pasos de Entrenamiento' }},
                    shapes: [{{
                        type: 'line',
                        xref: 'paper', x0: 0, x1: 1,
                        yref: 'y', y0: 0.8, y1: 0.8,
                        line: {{ color: 'red', width: 2, dash: 'dash' }}
                    }}],
                    legend: {{ orientation: 'h', y: 1.1 }}
                }});
            }}
        </script>
    </body>
    </html>
    """
    
    with open('analysis/vlmrl_elite_as_td_only.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("\n" + "="*65)
    print("✅ DASHBOARD ELITE (AS & TD) GENERADO")
    print(f"Archivo: {os.path.abspath('analysis/vlmrl_elite_as_td_only.html')}")
    print("="*65)

if __name__ == "__main__":
    generate_full_evaluation_dashboard()