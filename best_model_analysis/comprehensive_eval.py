import os
import pandas as pd
import numpy as np
import glob
import argparse
import sys

# Configuration for target values (can be adjusted)
TARGET_SPEED = 25.0  # km/h (Typical CARLA target)

def calculate_holistic_score(file_path):
    """
    Calculates a comprehensive score based on ALL available columns.
    Returns a dictionary of metrics and the final score.
    """
    try:
        df = pd.read_csv(file_path)
        
        # --- 1. SAFETY (CRITICAL) ---
        # Collisions: Any collision is a massive penalty.
        # We look at 'collision_speed'. If sum > 0, there was a crash.
        total_collision_intensity = df['collision_speed'].sum()
        
        # Lane Keeping: Max deviation is critical (going off-road).
        max_center_dev = df['center_dev'].max()
        mean_center_dev = df['center_dev'].mean()
        
        # --- 2. EFFICIENCY ---
        # Speed: We want to be close to target speed, not too slow, not too fast.
        # Penalty for deviation from target speed (only when not stopped).
        # We filter out initial stop (speed < 1) to avoid penalizing starting.
        moving_df = df[df['speed'] > 1.0]
        if not moving_df.empty:
            speed_dev = np.abs(moving_df['speed'] - TARGET_SPEED).mean()
        else:
            speed_dev = TARGET_SPEED # Penalize if it never moves
            
        # Progress: Distance traveled
        total_distance = df['distance'].iloc[-1] if not df.empty else 0
        
        # Routes Completed: The ultimate success metric per episode
        # We take the mean across episodes in this file (usually 1 file = 1 model's run in 1 scenario)
        # But wait, the raw csv contains ALL episodes for that model.
        # We need to aggregate per episode first?
        # The user wants the BEST MODEL. The raw csv contains multiple episodes (e.g. 0-9).
        # We should average the performance across all test episodes for this model.
        
        avg_routes_completed = df.groupby('episode')['routes_completed'].max().mean()
        
        # --- 3. COMFORT / STABILITY ---
        # Steer Noise: Jerkiness in steering
        steer_clean = df['steer'].dropna()
        steer_noise = np.mean(np.abs(np.diff(steer_clean))) if len(steer_clean) > 1 else 0
        
        # Throttle Noise: Jerkiness in acceleration
        throttle_clean = df['throttle'].dropna()
        throttle_noise = np.mean(np.abs(np.diff(throttle_clean))) if len(throttle_clean) > 1 else 0
        
        # --- 4. TRACKING ---
        # Heading Error: angle_next_waypoint
        # We want this to be low (pointing to the road).
        heading_error = np.abs(df['angle_next_waypoint']).mean()
        
        # --- SCORING FORMULA ---
        # We normalize or weight these to get a final "Cost" (Lower is better).
        
        # Weights (Arbitrary but logical based on driving priorities)
        w_collision = 1000.0  # Huge penalty for crashing
        w_success = 500.0     # Huge reward for finishing (negative cost)
        w_safety = 10.0       # Lane deviation
        w_comfort = 20.0      # Steer/Throttle noise
        w_tracking = 1.0      # Heading error
        w_speed = 0.5         # Speed deviation
        
        # Score Calculation
        # We invert "Routes Completed" to make it a Cost (1.0 -> 0.0, 0.0 -> 1.0)
        success_cost = (1.0 - avg_routes_completed) * w_success
        
        collision_cost = total_collision_intensity * w_collision
        
        safety_cost = (max_center_dev * 2.0 + mean_center_dev) * w_safety
        
        comfort_cost = (steer_noise + throttle_noise) * w_comfort
        
        tracking_cost = heading_error * w_tracking
        
        speed_cost = speed_dev * w_speed
        
        final_score = success_cost + collision_cost + safety_cost + comfort_cost + tracking_cost + speed_cost
        
        return {
            'score': final_score,
            'success_rate': avg_routes_completed,
            'collisions': total_collision_intensity,
            'max_dev': max_center_dev,
            'steer_noise': steer_noise,
            'speed_dev': speed_dev,
            'distance': total_distance
        }

    except Exception as e:
        return None

def analyze_all_scenarios(run_dir):
    # Find all subdirectories that look like evaluation folders
    # They usually start with "eval"
    subdirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)) and d.startswith("eval")]
    subdirs.sort()
    
    print(f"Found {len(subdirs)} evaluation scenarios: {', '.join(subdirs)}\n")
    
    overall_best_models = {}
    
    for scenario in subdirs:
        scenario_path = os.path.join(run_dir, scenario)
        csv_files = glob.glob(os.path.join(scenario_path, "*_eval.csv"))
        csv_files = [f for f in csv_files if "summary" not in f]
        
        if not csv_files:
            continue
            
        print(f"--- Analyzing Scenario: {scenario} ({len(csv_files)} models) ---")
        
        scenario_results = []
        
        for f in csv_files:
            filename = os.path.basename(f)
            try:
                steps = int(filename.split('_')[1])
            except:
                continue
                
            metrics = calculate_holistic_score(f)
            if metrics:
                metrics['steps'] = steps
                metrics['model'] = filename
                scenario_results.append(metrics)
        
        if not scenario_results:
            print("  No valid data found.")
            continue
            
        # Find winner for this scenario
        df = pd.DataFrame(scenario_results)
        best_model = df.sort_values(by='score', ascending=True).iloc[0]
        
        print(f"  🏆 Winner: Model {int(best_model['steps'])} steps")
        print(f"     Score: {best_model['score']:.2f} (Lower is better)")
        print(f"     Success: {best_model['success_rate']*100:.1f}% | Max Dev: {best_model['max_dev']:.2f}m | Collisions: {best_model['collisions']:.2f}")
        
        overall_best_models[scenario] = int(best_model['steps'])
        
    # Summary
    print("\n" + "="*60)
    print(" SUMMARY OF BEST MODELS PER SCENARIO")
    print("="*60)
    for scenario, steps in overall_best_models.items():
        print(f"{scenario:<25} -> Model {steps} steps")
        
    # Find the "Universal" best model (most frequent winner)
    from collections import Counter
    if overall_best_models:
        winners = list(overall_best_models.values())
        most_common = Counter(winners).most_common(1)
        print("-" * 60)
        print(f"🌟 UNIVERSAL CHAMPION: Model {most_common[0][0]} steps (Won {most_common[0][1]} scenarios)")
        print("-" * 60)

if __name__ == "__main__":
    #default_path = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20251027_081939_idvlm_rl"
    default_path = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=default_path)
    args = parser.parse_args()
    
    analyze_all_scenarios(args.run_dir)
