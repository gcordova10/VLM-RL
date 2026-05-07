import os
import pandas as pd
import numpy as np
import glob
import argparse
import sys

def calculate_stability_metrics(file_path):
    """
    Calculates stability metrics from a raw evaluation CSV.
    Returns None if the file cannot be read.
    """
    try:
        df = pd.read_csv(file_path)
        
        # 1. Steer Smoothness (Comfort)
        # Calculate the mean absolute difference between consecutive steer values.
        # Lower is smoother.
        steer_clean = df['steer'].dropna()
        if len(steer_clean) > 1:
            steer_diff = np.abs(np.diff(steer_clean))
            steer_noise = np.mean(steer_diff)
        else:
            steer_noise = 0.0
        
        # 2. Safety Critical (Center Deviation)
        # The maximum deviation from the center of the lane.
        # Lower is safer.
        max_center_dev = df['center_dev'].max()
        
        # 3. Trajectory Precision
        # The average deviation from the center.
        mean_center_dev = df['center_dev'].mean()
        
        return {
            'steer_noise': steer_noise,
            'max_center_dev': max_center_dev,
            'mean_center_dev': mean_center_dev
        }
    except Exception as e:
        # print(f"Error reading {file_path}: {e}")
        return None

def get_success_rate(summary_path):
    """
    Reads the success rate from the summary CSV.
    """
    try:
        df = pd.read_csv(summary_path)
        total_row = df[df['episode'] == 'total']
        if not total_row.empty and 'success' in total_row.columns:
            return float(total_row['success'].values[0])
    except:
        pass
    return 0.0

def analyze_models(run_dir):
    eval_dir = os.path.join(run_dir, "eval")
    if not os.path.exists(eval_dir):
        print(f"Error: Directory not found: {eval_dir}")
        return

    # Get all raw eval csvs (excluding summaries)
    csv_files = glob.glob(os.path.join(eval_dir, "*_eval.csv"))
    csv_files = [f for f in csv_files if "summary" not in f]
    
    results = []
    
    print(f"Analyzing {len(csv_files)} models in {eval_dir}...")
    print("Calculating metrics (this may take a moment)...")
    
    for f in csv_files:
        filename = os.path.basename(f)
        try:
            # Extract step count: model_10000_steps_eval.csv
            steps = int(filename.split('_')[1])
        except:
            continue
            
        # Calculate visual/stability metrics from raw data
        metrics = calculate_stability_metrics(f)
        
        # Get success rate from the corresponding summary file
        summary_filename = filename.replace(".csv", "_summary.csv")
        summary_path = os.path.join(eval_dir, summary_filename)
        success_rate = get_success_rate(summary_path)
        
        if metrics:
            metrics['steps'] = steps
            metrics['success_rate'] = success_rate
            
            # --- SCORING FORMULA ---
            # We want a single score to rank them. Lower is better.
            # 1. Safety is paramount: Weight Max Dev heavily.
            # 2. Comfort is secondary: Weight Steer Noise.
            # 3. Precision is tertiary: Weight Mean Dev.
            #
            # However, we MUST penalize models with low success rate.
            # If success < 1.0, add a huge penalty.
            
            base_score = (metrics['max_center_dev'] * 2.0) + (metrics['steer_noise'] * 10.0) + (metrics['mean_center_dev'] * 1.0)
            
            if success_rate < 1.0:
                penalty = (1.0 - success_rate) * 100.0 # Huge penalty for failure
            else:
                penalty = 0.0
                
            metrics['final_score'] = base_score + penalty
            results.append(metrics)
            
    if not results:
        print("No valid models found.")
        return

    df_res = pd.DataFrame(results)
    
    # Sort by Final Score (Ascending -> Lower is better)
    df_sorted = df_res.sort_values(by='final_score', ascending=True)
    
    print("\n" + "="*80)
    print(" FINAL RANKING (Based on Safety, Stability & Success)")
    print("="*80)
    print(f"{'Rank':<5} {'Steps':<10} {'Success':<10} {'Max Dev(m)':<12} {'Steer Noise':<12} {'Score (Lower=Better)':<20}")
    print("-" * 80)
    
    for i in range(min(10, len(df_sorted))):
        row = df_sorted.iloc[i]
        print(f"{i+1:<5} {int(row['steps']):<10} {row['success_rate']:<10.2f} {row['max_center_dev']:<12.4f} {row['steer_noise']:<12.4f} {row['final_score']:<20.4f}")
        
    best_model = df_sorted.iloc[0]
    print("\n" + "="*80)
    print(f"🏆 WINNER: Model {int(best_model['steps'])} steps")
    print(f"   - Success Rate: {best_model['success_rate']*100:.0f}%")
    print(f"   - Max Deviation: {best_model['max_center_dev']:.4f} m (Safety)")
    print(f"   - Steer Noise:   {best_model['steer_noise']:.4f} (Comfort)")
    print("="*80)

if __name__ == "__main__":
    # Default path if not provided
    #default_path = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20251027_081939_idvlm_rl"
    default_path = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl"

    parser = argparse.ArgumentParser(description="Find the best RL model based on stability and success.")
    parser.add_argument("--run_dir", type=str, default=default_path, help="Path to the tensorboard run directory")
    
    args = parser.parse_args()
    
    analyze_models(args.run_dir)
