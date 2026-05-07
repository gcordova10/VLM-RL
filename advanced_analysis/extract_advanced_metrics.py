import os
import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
import json
import sys

def extract_metrics(base_eval_dir, output_path):
    eval_folders = [d for d in os.listdir(base_eval_dir) if os.path.isdir(os.path.join(base_eval_dir, d)) and d.startswith('eval')]
    
    advanced_data = []

    for folder in eval_folders:
        eval_dir = os.path.join(base_eval_dir, folder)
        raw_files = sorted(glob.glob(os.path.join(eval_dir, 'model_*_steps_eval.csv')))
        
        print(f"Processing {len(raw_files)} files in {folder}...")
        for f in tqdm(raw_files):
            try:
                filename = os.path.basename(f)
                steps = int(filename.split('_')[1])
                
                df = pd.read_csv(f)
                # Filter out "route" metadata rows
                df_clean = df[df['model_id'] != 'route'].copy()
                
                if df_clean.empty:
                    continue
                    
                # Convert numeric columns
                numeric_cols = ['speed', 'center_dev', 'steer', 'throttle', 'collision_speed']
                for col in numeric_cols:
                    if col in df_clean.columns:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Group by episode to get per-episode metrics first
                episodes = df_clean.groupby('episode')
                
                ep_metrics = []
                for name, group in episodes:
                    # Basic
                    avg_speed = group['speed'].mean() if 'speed' in group.columns else 0
                    avg_center_dev = group['center_dev'].mean() if 'center_dev' in group.columns else 0
                    max_center_dev = group['center_dev'].max() if 'center_dev' in group.columns else 0
                    
                    # Quality: Jerk (Stability) - difference between consecutive steps
                    steer_diff = group['steer'].diff().abs().mean() if 'steer' in group.columns else 0
                    throttle_diff = group['throttle'].diff().abs().mean() if 'throttle' in group.columns else 0
                    
                    # Safety
                    collisions = (group['collision_speed'] > 0).any() if 'collision_speed' in group.columns else False
                    
                    ep_metrics.append({
                        'avg_speed': avg_speed,
                        'avg_center_dev': avg_center_dev,
                        'max_center_dev': max_center_dev,
                        'steer_stability': steer_diff,
                        'throttle_stability': throttle_diff,
                        'has_collision': int(collisions)
                    })
                
                # Aggregate episode metrics for this model
                ep_df = pd.DataFrame(ep_metrics)
                
                model_metrics = {
                    'scenario': folder,
                    'steps': steps,
                    'avg_speed': ep_df['avg_speed'].mean(),
                    'avg_center_dev': ep_df['avg_center_dev'].mean(),
                    'max_center_dev_mean': ep_df['max_center_dev'].mean(),
                    'steer_stability': ep_df['steer_stability'].mean(),
                    'throttle_stability': ep_df['throttle_stability'].mean(),
                    'collision_rate': ep_df['has_collision'].mean(),
                }
                
                # Get SR and RC from summary file
                summary_f = f.replace('.csv', '_summary.csv')
                if os.path.exists(summary_f):
                    sum_df = pd.read_csv(summary_f)
                    total_row = sum_df[sum_df['episode'] == 'total']
                    if not total_row.empty:
                        model_metrics['sr'] = float(total_row['success'].values[0])
                        model_metrics['rc'] = float(total_row['routes_completed'].values[0])
                
                advanced_data.append(model_metrics)
                
            except Exception as e:
                # print(f"Error processing {f}: {e}")
                continue

    # Asegurar que el directorio de salida existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w') as out:
        json.dump(advanced_data, out, indent=2)
    print(f"Saved advanced metrics to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 advanced_analysis/extract_advanced_metrics.py <BASE_EVAL_DIR> <OUTPUT_JSON>")
        sys.exit(1)
        
    extract_metrics(sys.argv[1], sys.argv[2])
