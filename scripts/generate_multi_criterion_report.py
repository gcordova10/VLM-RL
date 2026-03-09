import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def create_radar_plot():
    # Data and Descriptions
    criteria_info = {
        '1. Balanced (General)': {
            'obj': 'Encontrar el punto medio ideal entre completar la ruta y hacerlo de forma segura.',
            'philosophy': 'No sirve de nada llegar rápido si chocas mucho, ni ser muy seguro si nunca llegas al destino. Premia un RC alto con desviación mínima.',
            'metrics': ['Routes Completed', 'CPM (inv)', 'Avg Center Dev (inv)', 'Smoothness (inv)'],
            'baseline': [9.6553, 1/1.3307, 1/0.0994, 1/0.5000],
            'ours': [8.1262, 1/2.6616, 1/0.0585, 1/3.3588],
            'steps': ('470k', '410k')
        },
        '2. Fluid Driving (State)': {
            'obj': 'Evaluar la estabilidad del agente a largo plazo y la calidad del "estado" de conducción.',
            'philosophy': 'Se enfoca en la capacidad de mantenerse en el carril. Un Coll Interval alto indica conducción prolongada sin errores críticos.',
            'metrics': ['Routes Completed', 'Avg Center Dev (inv)', 'Coll Interval'],
            'baseline': [8.0025, 1/0.0708, 30037.0],
            'ours': [8.1262, 1/0.0585, 16326.0],
            'steps': ('990k', '410k')
        },
        '3. High Speed (Efficacy)': {
            'obj': 'Maximizar la velocidad de transporte manteniendo un nivel aceptable de riesgo.',
            'philosophy': 'Para modelos "agresivos" pero competentes. Valora la velocidad y que los impactos sean leves (baja Coll Speed).',
            'metrics': ['Routes Completed', 'Avg Speed', 'CPM (inv)', 'Coll Speed (inv)'],
            'baseline': [9.6553, 18.7816, 1/1.3307, 1/0.0969],
            'ours': [8.4534, 19.5536, 1/4.9319, 1/0.0371],
            'steps': ('470k', '820k')
        },
        '4. Learning Consistency (Resilience)': {
            'obj': 'Medir la madurez y robustez del entrenamiento.',
            'philosophy': 'Utiliza la recompensa base (GT) como indicador de éxito puro. Mide si el éxito es repetible y resiliente.',
            'metrics': ['Coll Interval', 'GT Reward'],
            'baseline': [30037.0, 1635.3667],
            'ours': [18025.0, 1853.1320],
            'steps': ('990k', '510k')
        },
        '5. Technical Fidelity (VLM-RL Paper)': {
            'obj': 'Comparar resultados bajo el estándar estricto del paper original.',
            'philosophy': 'El criterio más riguroso. Exige comportamiento humano: suave, preciso y con alto Total Reward.',
            'metrics': ['Collision Rate (inv)', 'Routes Completed', 'Total Reward', 'Smoothness (inv)', 'Avg Center Dev (inv)'],
            'baseline': [1/0.74, 9.6553, 3621.7766, 1/0.50, 1/0.0994],
            'ours': [1/0.50, 8.0035, 3388.7502, 1/9.1470, 1/0.2153],
            'steps': ('470k', '100k')
        }
    }

    # Normalize data
    normalized_data = {}
    for crit, values in criteria_info.items():
        norm_baseline = []
        norm_ours = []
        for b, o in zip(values['baseline'], values['ours']):
            max_val = max(b, o)
            norm_baseline.append((b / max_val) * 100)
            norm_ours.append((o / max_val) * 100)
        normalized_data[crit] = {'baseline': norm_baseline, 'ours': norm_ours}

    # Create figure with more spacing
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}],
               [{'type': 'polar'}, {'type': 'polar'}, None]],
        subplot_titles=[f"<b>{k}</b>" for k in criteria_info.keys()],
        horizontal_spacing=0.15,
        vertical_spacing=0.2
    )

    colors = {'Baseline': '#636EFA', 'Ours': '#00CC96'}
    
    for i, (crit, info) in enumerate(criteria_info.items()):
        row = (i // 3) + 1
        col = (i % 3) + 1
        fig.add_trace(go.Scatterpolar(
            r=normalized_data[crit]['baseline'], theta=info['metrics'],
            fill='toself', name=f'Baseline (Step {info["steps"][0]})',
            line_color=colors['Baseline'], legendgroup='Baseline', showlegend=(i == 0)
        ), row=row, col=col)
        fig.add_trace(go.Scatterpolar(
            r=normalized_data[crit]['ours'], theta=info['metrics'],
            fill='toself', name=f'Ours (Step {info["steps"][1]})',
            line_color=colors['Ours'], legendgroup='Ours', showlegend=(i == 0)
        ), row=row, col=col)

    fig.update_layout(
        template="plotly_dark", height=1000, width=1400,
        margin=dict(t=150, b=100, l=50, r=50),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )

    # Generate HTML Parts
    criteria_html = "".join([f"""
        <div class='criterion-box'>
            <h3>{name}</h3>
            <p><b>Objetivo:</b> {info['obj']}</p>
            <p><b>Filosofía:</b> {info['philosophy']}</p>
        </div>
    """ for name, info in criteria_info.items()])

    metrics_html = """
        <div class='section'>
            <h2>Diccionario de Métricas</h2>
            <div class='metrics-grid'>
                <div class='metric-cat'>
                    <h4>1. Éxito (Rendimiento)</h4>
                    <ul>
                        <li><b>RC (Routes Completed):</b> % de rutas terminadas con éxito. Indica si el agente "sabe a dónde va".</li>
                        <li><b>Avg Speed (AS):</b> Velocidad media (km/h). Mide la eficiencia temporal.</li>
                    </ul>
                </div>
                <div class='metric-cat'>
                    <h4>2. Seguridad (Críticas)</h4>
                    <ul>
                        <li><b>Collision Rate (CR):</b> Probabilidad de choque por episodio.</li>
                        <li><b>CPM (Collisions Per Mile):</b> Choques por milla. Refleja la densidad de fallos.</li>
                        <li><b>Coll Interval (ICT):</b> Tiempo/Pasos de conducción segura continua. Mide la robustez.</li>
                        <li><b>Coll Speed (CS):</b> Velocidad en el impacto. Evalúa la capacidad de frenado de emergencia.</li>
                    </ul>
                </div>
                <div class='metric-cat'>
                    <h4>3. Calidad (Fidelidad)</h4>
                    <ul>
                        <li><b>Avg Center Dev (ACD):</b> Desviación media del centro del carril (metros). Mide la precisión.</li>
                        <li><b>Smoothness:</b> Jitter o temblor del volante. Refleja la estabilidad del control humanoide.</li>
                    </ul>
                </div>
                <div class='metric-cat'>
                    <h4>4. Aprendizaje</h4>
                    <ul>
                        <li><b>Total Reward:</b> Recompensa acumulada (incluye guías VLM).</li>
                        <li><b>GT Reward:</b> Recompensa pura del entorno CARLA. Verifica el aprendizaje real.</li>
                    </ul>
                </div>
            </div>
        </div>
    """

    full_html = f"""
    <html>
    <head>
        <title>Multi-Criterion Comparative Report: VLM-RL</title>
        <style>
            body {{ background-color: #111; color: #eee; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #00CC96; border-bottom: 2px solid #333; padding-bottom: 10px; }}
            h3 {{ color: #636EFA; margin-bottom: 5px; }}
            .criterion-box {{ background: #222; padding: 15px; margin-bottom: 10px; border-radius: 8px; border-left: 5px solid #636EFA; }}
            .section {{ margin-top: 40px; padding: 20px; background: #1a1a1a; border-radius: 12px; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
            .metric-cat {{ background: #252525; padding: 15px; border-radius: 8px; }}
            ul {{ padding-left: 20px; }}
            li {{ margin-bottom: 10px; font-size: 0.9em; }}
            .chart-container {{ background: #1a1a1a; padding: 20px; border-radius: 12px; margin-bottom: 40px; }}
        </style>
    </head>
    <body>
        <h1>Multi-Criterion Comparative Report: VLM-RL</h1>
        <p>Este reporte analiza el rendimiento comparativo entre el <b>Baseline (RL Estándar)</b> y <b>Ours (+VLM Guidance)</b> bajo diferentes prismas operativos.</p>
        
        <div class='chart-container'>
            {fig.to_html(full_html=False, include_plotlyjs='cdn')}
        </div>

        <div class='section'>
            <h2>Descripción de los Criterios de Selección</h2>
            {criteria_html}
        </div>

        {metrics_html}
        
        <div style='margin-top: 50px; font-size: 0.8em; color: #666; text-align: center;'>
            Generado automáticamente por VLM-RL Analysis Suite | 2026
        </div>
    </body>
    </html>
    """

    output_path = "tensorboard_analysis/vlmrl_multi_criterion_radar.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"Report fully updated: {output_path}")

if __name__ == "__main__":
    create_radar_plot()
