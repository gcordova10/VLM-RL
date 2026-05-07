import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

SCENARIOS = [
    "eval", "evaldense", "evaldenseTown01", "evaldenseTown03", "evaldenseTown04", "evaldenseTown05",
    "evalempty", "evalemptyTown01", "evalemptyTown03", "evalemptyTown04", "evalemptyTown05",
    "evalTown01", "evalTown03", "evalTown04", "evalTown05"
]

STEPS = [i for i in range(10000, 1000001, 10000)]

def process_telemetry_file(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df_telemetry = df[df['model_id'] != 'route'].copy()
        cols = ['speed', 'center_dev', 'steer_smoothness', 'reward', 'collision_speed', 
                'collision_interval', 'routes_completed', 'CPM']
        for col in cols:
            if col in df_telemetry.columns:
                df_telemetry[col] = pd.to_numeric(df_telemetry[col], errors='coerce')
            else:
                df_telemetry[col] = np.nan

        episodes = df_telemetry['episode'].unique()
        ep_stats = []
        for ep in episodes:
            ep_df = df_telemetry[df_telemetry['episode'] == ep]
            moving_df = ep_df[ep_df['speed'] > 0.5]
            if moving_df.empty: continue
            ep_stats.append({
                'RC': ep_df['routes_completed'].max(),
                'CPM': ep_df['CPM'].iloc[-1] if 'CPM' in ep_df.columns else np.nan,
                'ACD': moving_df['center_dev'].mean() if 'center_dev' in moving_df.columns else np.nan,
                'Smoothness': moving_df['steer_smoothness'].mean() if 'steer_smoothness' in moving_df.columns else np.nan,
                'AS': ep_df['speed'].mean(),
                'ICT': ep_df['collision_interval'].max() if 'collision_interval' in ep_df.columns else np.nan,
                'CS': ep_df['collision_speed'].max() if 'collision_speed' in ep_df.columns else np.nan,
                'Has_Collision': 1 if ep_df['collision_speed'].max() > 0 else 0,
                'Total_Reward': ep_df['reward'].sum()
            })
        if not ep_stats: return None
        return pd.DataFrame(ep_stats).mean().to_dict()
    except: return None

def aggregate_model_performance(base_path):
    aggregated_results = []
    folder_name = os.path.basename(base_path.rstrip('/'))
    for step in tqdm(STEPS, desc=f"Analyzing {folder_name}"):
        step_metrics = []
        for scene in SCENARIOS:
            csv_path = os.path.join(base_path, scene, f"model_{step}_steps_eval.csv")
            if os.path.exists(csv_path):
                metrics = process_telemetry_file(csv_path)
                if metrics: step_metrics.append(metrics)
        if step_metrics:
            step_avg = pd.DataFrame(step_metrics).mean().to_dict()
            step_avg['Step'] = step
            aggregated_results.append(step_avg)
    return pd.DataFrame(aggregated_results)

def calculate_universal_bests(df):
    if df.empty: return {}
    def n(s, invert=False): 
        if s.isna().all() or (s.dropna().max() == s.dropna().min()): 
            return pd.Series(0.5, index=s.index)
        norm = (s - s.min()) / (s.max() - s.min() + 1e-6)
        if invert: norm = 1.0 - norm
        return norm.fillna(0.5)
    
    rc = n(df['RC']); cpm_inv = n(df['CPM'], invert=True); acd_inv = n(df['ACD'], invert=True)
    smooth_inv = n(df['Smoothness'], invert=True); as_speed = n(df['AS']); cs_inv = n(df['CS'], invert=True)
    ict = n(df['ICT']); rew = n(df['Total_Reward']); cr_inv = n(df['Has_Collision'], invert=True)
    results = {}
    
    def print_universal_best(name, series):
        step = int(series['Step'])
        m_str = f"RC: {series.get('RC', 0):.2f} | CPM: {series.get('CPM', 0):.2f} | ACD: {series.get('ACD', 0):.4f} | ICT: {series.get('ICT', 0):.0f} | Smooth: {series.get('Smoothness', 0):.4f}"
        print(f"  > Universal Best {name:25} : Step {step:7} | {m_str}")

    results["Balanced (Universal)"] = df.iloc[((rc+0.1)*(cpm_inv+0.1)*(acd_inv+0.1)*(smooth_inv+0.1)).idxmax()]
    print_universal_best("Balanced (Universal)", results["Balanced (Universal)"])

    results["Fluid Driving (State)"] = df.iloc[((rc+0.1)*(acd_inv+0.1)*(ict+0.1)).idxmax()]
    print_universal_best("Fluid Driving (State)", results["Fluid Driving (State)"])

    results["High Speed (Efficiency)"] = df.iloc[((rc+0.1)*(as_speed+0.1)*(cpm_inv+0.1)*(cs_inv+0.1)).idxmax()]
    print_universal_best("High Speed (Efficiency)", results["High Speed (Efficiency)"])

    results["Consistency (Resilience)"] = df.iloc[((ict+0.1)*(rew+0.1)).idxmax()]
    print_universal_best("Consistency (Resilience)", results["Consistency (Resilience)"])

    results["Technical Fidelity (VLM-RL Paper)"] = df.iloc[(0.4*cr_inv + 0.3*rc + 0.15*rew + 0.1*smooth_inv + 0.05*acd_inv).idxmax()]
    print_universal_best("Technical Fidelity (VLM-RL Paper)", results["Technical Fidelity (VLM-RL Paper)"])
    
    return results

def generate_html_report(bests_b, bests_o, output_file):
    html = "<html><head><style>body{font-family:sans-serif;background:#121212;color:#eee;padding:40px;} .card{background:#1e1e1e;padding:20px;border-radius:10px;margin-bottom:30px;border-top:4px solid #3498db;} .baseline{border-top-color:#e74c3c;} .ours{border-top-color:#2ecc71;} h1,h2{text-align:center;} table{width:100%;border-collapse:collapse;} td,th{padding:10px;border:1px solid #333;text-align:right;} .metric{text-align:left;font-weight:bold;color:#3498db;}</style></head><body>"
    html += f"<h1>UNIVERSAL GENERALIZATION RANKING</h1>"
    for crit in bests_o.keys():
        o = bests_o[crit]; b = bests_b.get(crit, None)
        html += f"<h2>Criterion: {crit}</h2><div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;'>"
        for label, data, cls in [("BASELINE", b, "baseline"), ("OURS (+CLG-Smooth)", o, "ours")]:
            if data is not None:
                html += f"<div class='card {cls}'><h3>{label} - Step {int(data['Step'])}</h3><table>"
                for k,v in data.items():
                    if k != 'Step': html += f"<tr><td class='metric'>{k}</td><td>{v:.4f}</td></tr>"
                html += "</table></div>"
            else: html += f"<div class='card {cls}'><h3>{label} - No hay datos</h3></div>"
        html += "</div>"
    html += "</body></html>"
    with open(output_file, "w") as f: f.write(html)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, help="Ruta a la carpeta del Baseline")
    parser.add_argument("--ours", type=str, required=True, help="Ruta a la carpeta de Ours")
    parser.add_argument("--output", type=str, default="ranking_universal_generalizacion.html", help="Nombre del archivo de salida")
    args = parser.parse_args()

    df_b = aggregate_model_performance(args.baseline)
    df_o = aggregate_model_performance(args.ours)
    
    # Ensure Smoothness exists and has a neutral value (0.5) ONLY IN BASELINE
    if not df_b.empty:
        if "Smoothness" not in df_b.columns:
            df_b["Smoothness"] = 0.5
        else:
            df_b["Smoothness"] = np.where(df_b["Smoothness"].isna() | (df_b["Smoothness"] < 0.001), 0.5, df_b["Smoothness"])

    print("\n--- UNIVERSAL BASELINE RESULTS ---")
    b_results = calculate_universal_bests(df_b)
    
    print("\n--- UNIVERSAL OURS RESULTS ---")
    o_results = calculate_universal_bests(df_o)
    
    generate_html_report(b_results, o_results, args.output)
    print(f"\n✅ Report generated successfully: {args.output}")
