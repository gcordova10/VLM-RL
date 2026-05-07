import os
import sys
import pandas as pd
import json
from tensorboard.backend.event_processing import event_accumulator

def extract_tb_value(ea, tag, step):
    if tag not in ea.Tags()['scalars']:
        return 0.0
    scalars = ea.Scalars(tag)
    # Find closest step
    closest_val = 0.0
    min_diff = float('inf')
    for s in scalars:
        diff = abs(s.step - step)
        if diff < min_diff:
            min_diff = diff
            closest_val = s.value
        if diff == 0: break
    return closest_val

def get_run_metrics(log_dir, steps_to_extract):
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        return {}
    
    event_file = os.path.join(log_dir, event_files[0])
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    tags = [
        'custom/routes_completed', 'custom/CPM', 'custom/avg_center_dev', 
        'custom/collision_interval', 'custom/avg_speed', 'custom/collision_speed', 
        'custom/collision_rate', 'custom/total_reward', 'rollout/ep_gt_rew_mean'
    ]
    
    results = {}
    for step in steps_to_extract:
        step_metrics = {}
        for tag in tags:
            val = extract_tb_value(ea, tag, step)
            key = tag.split('/')[-1]
            step_metrics[key] = val
        results[step] = step_metrics
        
    return results

def main():
    baseline_dir = '/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl'
    ours_dir = '/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl'
    
    # Steps from the original report
    # BT: 470k, 990k
    # OT: 100k, 410k, 510k, 820k
    baseline_steps = [470000, 990000]
    ours_steps = [100000, 410000, 510000, 820000]
    
    print("Extracting Baseline metrics...")
    baseline_metrics = get_run_metrics(baseline_dir, baseline_steps)
    print("Extracting Ours metrics...")
    ours_metrics = get_run_metrics(ours_dir, ours_steps)
    
    with open("data/radar_metrics_ours.json", "w") as f:
        json.dump({
            "baseline": baseline_metrics,
            "ours": ours_metrics
        }, f, indent=2)
    print("Done! Metrics saved to data/radar_metrics_ours.json")

if __name__ == "__main__":
    main()
