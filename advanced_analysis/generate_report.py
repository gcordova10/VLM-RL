import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

def generate_html_report(input_json, output_html, video_base_path):
    if not os.path.exists(input_json):
        print(f"Error: {input_json} not found")
        return

    with open(input_json, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # 1. Global Pareto Frontier (Efficiency vs Safety)
    # Filter only models with some success to avoid noise
    df_pareto = df[df['sr'] > 0.5].copy()
    if not df_pareto.empty:
        fig_pareto = px.scatter(
            df_pareto, 
            x="avg_speed", 
            y="avg_center_dev",
            color="sr",
            size="rc",
            hover_data=["steps", "scenario"],
            title="Pareto Frontier: Velocidad vs Desviación (Modelos con SR > 0.5)",
            labels={"avg_speed": "Velocidad Media (Eficiencia)", "avg_center_dev": "Desviación Media (Seguridad)"},
            template="plotly_dark"
        )
        # We want low deviation and high speed
        fig_pareto.update_yaxes(autorange="reversed") 
    else:
        fig_pareto = go.Figure()

    # 2. Stability vs Training Time
    stability_data = df.groupby('steps')[['steer_stability', 'throttle_stability']].mean().reset_index()
    fig_stability = px.line(
        stability_data,
        x="steps",
        y=["steer_stability", "throttle_stability"],
        title="Evolución de la Estabilidad (Jerk) del Control",
        labels={"value": "Inestabilidad (Mean Abs Diff)", "steps": "Pasos de Entrenamiento"},
        template="plotly_dark"
    )

    # 3. Robustness Heatmap (Scenario vs Steps)
    pivot_sr = df.pivot(index="scenario", columns="steps", values="sr")
    fig_heatmap = px.imshow(
        pivot_sr,
        title="Heatmap de Robustez: Success Rate por Escenario y Checkpoint",
        labels=dict(x="Steps", y="Scenario", color="Success Rate"),
        template="plotly_dark",
        aspect="auto"
    )

    # 4. Composite Scoring (The "Best" Model)
    # Normalize metrics to 0-1
    def normalize(series, reverse=False):
        if series.max() == series.min():
            return series * 0 + (1.0 if not reverse else 0.0)
        if reverse:
            return (series.max() - series) / (series.max() - series.min() + 1e-6)
        return (series - series.min()) / (series.max() - series.min() + 1e-6)

    df['score_speed'] = normalize(df['avg_speed'])
    df['score_safety'] = normalize(df['avg_center_dev'], reverse=True)
    df['score_stability'] = normalize(df['steer_stability'], reverse=True)
    df['score_sr'] = df['sr']
    
    # Composite score: 40% Success, 20% Speed, 20% Safety, 20% Stability
    df['composite_score'] = (df['score_sr'] * 0.4 + 
                            df['score_speed'] * 0.2 + 
                            df['score_safety'] * 0.2 + 
                            df['score_stability'] * 0.2)
    
    best_models = df.groupby('steps')['composite_score'].mean().sort_values(ascending=False).head(10)
    
    fig_best = px.bar(
        best_models.reset_index(),
        x="steps",
        y="composite_score",
        title="Top 10 Checkpoints (Puntaje Compuesto Global)",
        template="plotly_dark"
    )

    # Combined HTML
    html_content = f"""
    <html>
    <head>
        <title>VLM-RL Advanced Evaluation Dashboard</title>
        <style>
            body {{ background-color: #111; color: #eee; font-family: sans-serif; margin: 20px; }}
            .chart {{ margin-bottom: 50px; background: #222; padding: 15px; border-radius: 8px; }}
            h1 {{ color: #00ccff; }}
            .best-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            .best-table th, .best-table td {{ border: 1px solid #444; padding: 10px; text-align: left; }}
            .best-table th {{ background: #333; }}
        </style>
    </head>
    <body>
        <h1>VLM-RL: Análisis Avanzado de Checkpoints</h1>
        <p>Comparativa de todos los escenarios por checkpoint.</p>
        
        <div class="chart">{fig_best.to_html(full_html=False, include_plotlyjs='cdn')}</div>
        <div class="chart">{fig_pareto.to_html(full_html=False, include_plotlyjs='cdn')}</div>
        <div class="chart">{fig_heatmap.to_html(full_html=False, include_plotlyjs='cdn')}</div>
        <div class="chart">{fig_stability.to_html(full_html=False, include_plotlyjs='cdn')}</div>
        
        <h2>Top 5 Checkpoints - DAFO Simplificado</h2>
        <table class="best-table">
            <tr>
                <th>Checkpoint (Steps)</th>
                <th>Global Score</th>
                <th>Fortalezas</th>
                <th>Debilidades</th>
                <th>Video Recomendado (Town02)</th>
            </tr>
            {generate_dafo_rows(df, video_base_path)}
        </table>
        <p><i>Nota: Los videos recomendados se encuentran en el directorio 'eval/' de los logs de tensorboard.</i></p>
    </body>
    </html>
    """
    
    # Asegurar que el directorio de salida existe
    os.makedirs(os.path.dirname(output_html) if os.path.dirname(output_html) else '.', exist_ok=True)

    with open(output_html, 'w') as f:
        f.write(html_content)
    print(f"Report generated: {output_html}")

def generate_dafo_rows(df, video_base_path):
    top_steps = df.groupby('steps')['composite_score'].mean().sort_values(ascending=False).head(5).index
    rows = ""
    for step in top_steps:
        model_data = df[df['steps'] == step]
        score = model_data['composite_score'].mean()
        
        # Determine strengths/weaknesses
        speed = model_data['avg_speed'].mean()
        safety = model_data['avg_center_dev'].mean()
        stability = model_data['steer_stability'].mean()
        
        strengths = []
        weaknesses = []
        
        if speed > df['avg_speed'].mean(): strengths.append("Alta Velocidad")
        else: weaknesses.append("Lento")
        
        if safety < df['avg_center_dev'].mean(): strengths.append("Precisión en Carril")
        else: weaknesses.append("Oscilación Lateral")
        
        if stability < df['steer_stability'].mean(): strengths.append("Conducción Suave")
        else: weaknesses.append("Control Nervioso")
        
        video_path = os.path.join(video_base_path, f"eval/model_{step}_steps_eval.avi")
        
        rows += f"""
        <tr>
            <td>{step}</td>
            <td>{score:.4f}</td>
            <td style='color: #44ff44;'>{", ".join(strengths)}</td>
            <td style='color: #ff4444;'>{", ".join(weaknesses)}</td>
            <td><code>{video_path}</code></td>
        </tr>
        """
    return rows

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 advanced_analysis/generate_report.py <INPUT_JSON> <OUTPUT_HTML> [VIDEO_BASE_PATH]")
        sys.exit(1)
    
    video_base = sys.argv[3] if len(sys.argv) > 3 else "."
    generate_html_report(sys.argv[1], sys.argv[2], video_base)
