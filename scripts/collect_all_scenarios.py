import os
import pandas as pd
import json
import glob
import re
import sys

def clean_val(val):
    if pd.isna(val): return 0.0
    s = str(val).strip()
    if not s: return 0.0
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    if matches:
        return float(matches[-1])
    return 0.0

def collect_data(base_path, output_name):
    scenarios = {
        "eval": "Town02",
        "evalempty": "emptyTown02",
        "evaldense": "denseTown02",
        "evalTown01": "Town01",
        "evalemptyTown01": "emptyTown01",
        "evaldenseTown01": "denseTown01",
        "evalTown03": "Town03",
        "evalemptyTown03": "emptyTown03",
        "evaldenseTown03": "denseTown03",
        "evalTown04": "Town04",
        "evalemptyTown04": "emptyTown04",
        "evaldenseTown04": "denseTown04",
        "evalTown05": "Town05",
        "evalemptyTown05": "emptyTown05",
        "evaldenseTown05": "denseTown05",
    }
    
    all_data = {}

    for folder, label in scenarios.items():
        full_path = os.path.join(base_path, folder)
        if not os.path.exists(full_path):
            continue
            
        print(f"Processing {label}...")
        csv_files = glob.glob(os.path.join(full_path, "*_eval_summary.csv"))
        scenario_results = []

        for f_path in csv_files:
            try:
                match = re.search(r"model_(\d+)_steps", os.path.basename(f_path))
                if not match: continue
                steps_val = int(match.group(1))
                df = pd.read_csv(f_path, dtype=object)
                total_row = df[df['episode'].astype(str).str.lower() == 'total']
                
                if not total_row.empty:
                    row = total_row.iloc[0]
                    res = {
                        "steps": steps_val,
                        "as": clean_val(row.get('speed_mean', 0)),
                        "rc": clean_val(row.get('routes_completed', 0)),
                        "td": clean_val(row.get('total_distance', 0)),
                        "cs": clean_val(row.get('collision_speed', 0)),
                        "sr": clean_val(row.get('success', 0))
                    }
                    scenario_results.append(res)
            except Exception as e:
                print(f"Error in {f_path}: {e}")

        scenario_results.sort(key=lambda x: x["steps"])
        all_data[label] = scenario_results

    output_path = os.path.join("data", output_name)
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Done! Generated {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 collect_all_scenarios.py <PATH_TO_TENSORBOARD_FOLDER> <OUTPUT_JSON_NAME>")
        sys.exit(1)
    
    collect_data(sys.argv[1], sys.argv[2])