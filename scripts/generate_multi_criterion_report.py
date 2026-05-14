import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def extract_tb_value(ea, tag, step):
    if tag not in ea.Tags()['scalars']: return 0.0
    scalars = ea.Scalars(tag)
    closest_val = 0.0
    min_diff = float('inf')
    for s in scalars:
        diff = abs(s.step - step)
        if diff < min_diff:
            min_diff = diff
            closest_val = s.value
        if diff == 0: break
    return closest_val

def get_smoothness(dq_data, step):
    step_data = [d['jitter'] for d in dq_data if d['steps'] == step]
    return sum(step_data) / len(step_data) if step_data else 0.1

def create_radar_plot(base_path, output_path):
    print(f"Loading data from: {base_path}")
    
    # 1. Load TensorBoard Data
    event_files = [f for f in os.listdir(base_path) if f.startswith('events.out.tfevents')]
    if not event_files:
        print("Error: No event files found.")
        return
    
    ea = event_accumulator.EventAccumulator(os.path.join(base_path, event_files[0]))
    ea.Reload()
    
    # 2. Load Quality Data (Jitter)
    # We assume the quality json for this specific run exists (we generated it before)
    dq_path = "data/driving_quality_20260212.json"
    with open(dq_path, "r") as f:
        dq_data = json.load(f)

    # 3. Baseline Data (Fixed for comparison - from the original paper/run)
    # These are the reference values from the 2025 run
    baseline_ref = {
        'rc': {470000: 9.6553, 990000: 8.0025},
        'cpm': {470000: 1.3307, 990000: 1.0},
        'acd': {470000: 0.0994, 990000: 0.0708},
        'sm': {470000: 0.50, 990000: 0.50},
        'ci': {470000: 10000, 990000: 30037.0},
        'as': {470000: 18.7816, 990000: 20.0},
        'cs': {470000: 0.0969, 990000: 0.1},
        'gr': {470000: 1500, 990000: 1635.3667},
        'tr': {470000: 3621.7766, 990000: 3500},
        'cr': {470000: 0.74, 990000: 0.5}
    }

    # Helper to get Ours metrics dynamically
    def get_ours(step):
        return {
            'rc': extract_tb_value(ea, 'custom/routes_completed', step),
            'cpm': extract_tb_value(ea, 'custom/CPM', step),
            'acd': extract_tb_value(ea, 'custom/avg_center_dev', step),
            'sm': get_smoothness(dq_data, step),
            'ci': extract_tb_value(ea, 'custom/collision_interval', step),
            'as': extract_tb_value(ea, 'custom/avg_speed', step),
            'cs': extract_tb_value(ea, 'custom/collision_speed', step),
            'gr': extract_tb_value(ea, 'rollout/ep_gt_rew_mean', step),
            'tr': extract_tb_value(ea, 'custom/total_reward', step),
            'cr': extract_tb_value(ea, 'custom/collision_rate', step)
        }

    # 4. Define Criteria using dynamic Ours data
    criteria_info = {
        '1. Balanced (General)': {
            'metrics': ['Routes Completed', 'CPM (inv)', 'Avg Center Dev (inv)', 'Smoothness (inv)'],
            'baseline': [baseline_ref['rc'][470000], 1/baseline_ref['cpm'][470000], 1/baseline_ref['acd'][470000], 1/baseline_ref['sm'][470000]],
            'ours_raw': lambda: [get_ours(410000)['rc'], 1/max(0.01, get_ours(410000)['cpm']), 1/max(0.01, get_ours(410000)['acd']), 1/max(0.01, get_ours(410000)['sm'])],
            'steps': ('470k', '410k')
        },
        '2. Fluid Driving (State)': {
            'metrics': ['Routes Completed', 'Avg Center Dev (inv)', 'Coll Interval'],
            'baseline': [baseline_ref['rc'][990000], 1/baseline_ref['acd'][990000], baseline_ref['ci'][990000]],
            'ours_raw': lambda: [get_ours(410000)['rc'], 1/max(0.01, get_ours(410000)['acd']), get_ours(410000)['ci']],
            'steps': ('990k', '410k')
        },
        '3. High Speed (Efficacy)': {
            'metrics': ['Routes Completed', 'Avg Speed', 'CPM (inv)', 'Coll Speed (inv)'],
            'baseline': [baseline_ref['rc'][470000], baseline_ref['as'][470000], 1/baseline_ref['cpm'][470000], 1/baseline_ref['cs'][470000]],
            'ours_raw': lambda: [get_ours(820000)['rc'], get_ours(820000)['as'], 1/max(0.01, get_ours(820000)['cpm']), 1/max(0.01, get_ours(820000)['cs'])],
            'steps': ('470k', '820k')
        },
        '4. Learning Consistency': {
            'metrics': ['Coll Interval', 'GT Reward'],
            'baseline': [baseline_ref['ci'][990000], baseline_ref['gr'][990000]],
            'ours_raw': lambda: [get_ours(510000)['ci'], get_ours(510000)['gr']],
            'steps': ('990k', '510k')
        },
        '5. Technical Fidelity': {
            'metrics': ['Collision Rate (inv)', 'Routes Completed', 'Total Reward', 'Smoothness (inv)', 'Avg Center Dev (inv)'],
            'baseline': [1/baseline_ref['cr'][470000], baseline_ref['rc'][470000], baseline_ref['tr'][470000], 1/baseline_ref['sm'][470000], 1/baseline_ref['acd'][470000]],
            'ours_raw': lambda: [1/max(0.01, get_ours(100000)['cr']), get_ours(100000)['rc'], get_ours(100000)['tr'], 1/max(0.01, get_ours(100000)['sm']), 1/max(0.01, get_ours(100000)['acd'])],
            'steps': ('470k', '100k')
        }
    }

    # 5. Process and Normalize
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}],
               [{'type': 'polar'}, {'type': 'polar'}, None]],
        subplot_titles=[f"<b>{k}</b>" for k in criteria_info.keys()],
        horizontal_spacing=0.15, vertical_spacing=0.2
    )

    colors = {'Baseline': '#636EFA', 'Ours': '#00CC96'}
    
    for i, (crit, info) in enumerate(criteria_info.items()):
        row, col = (i // 3) + 1, (i % 3) + 1
        ours_vals = info['ours_raw']()
        
        # Normalize
        norm_b, norm_o = [], []
        for b, o in zip(info['baseline'], ours_vals):
            mx = max(abs(b), abs(o), 1e-6)
            norm_b.append((b / mx) * 100)
            norm_o.append((o / mx) * 100)

        fig.add_trace(go.Scatterpolar(
            r=norm_b, theta=info['metrics'], fill='toself', name=f'Baseline ({info["steps"][0]})',
            line_color=colors['Baseline'], legendgroup='Baseline', showlegend=(i == 0)
        ), row=row, col=col)
        fig.add_trace(go.Scatterpolar(
            r=norm_o, theta=info['metrics'], fill='toself', name=f'Ours ({info["steps"][1]})',
            line_color=colors['Ours'], legendgroup='Ours', showlegend=(i == 0)
        ), row=row, col=col)

    fig.update_layout(
        template="plotly_dark", height=1000, width=1400,
        title=dict(text=f"VLM-RL RADAR: {os.path.basename(base_path)}", font=dict(size=22), x=0.5),
        margin=dict(t=150, b=100, l=50, r=50),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )

    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"Success! Dynamic report generated: {output_path}")

if __name__ == "__main__":
    #path = sys.argv[1] if len(sys.argv) > 1 else '/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl'
    #out = sys.argv[2] if len(sys.argv) > 2 else 'tensorboard_analysis/vlmrl_multi_criterion_radar_dynamic_20250930.html'
    path = sys.argv[1] if len(sys.argv) > 1 else '/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl'
    out = sys.argv[2] if len(sys.argv) > 2 else 'tensorboard_analysis/vlmrl_multi_criterion_radar_dynamic_20260212_ours.html'
    
    create_radar_plot(path, out)
