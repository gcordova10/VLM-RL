import os
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paths
baseline_path = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0"
ours_path = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"

# Metrics mapping with descriptions in English
metrics_info = {
    'RC (Routes Completed)': {
        'tag': 'custom/routes_completed',
        'desc': 'Percentage of routes successfully completed.'
    },
    'AS (Avg Speed)': {
        'tag': 'custom/avg_speed',
        'desc': 'Average vehicle speed (km/h).'
    },
    'CR (Collision Rate)': {
        'tag': 'custom/collision_rate',
        'desc': 'Rate of episodes ending in a collision.'
    },
    'CPM (Collisions Per Mile)': {
        'tag': 'custom/CPM',
        'desc': 'Frequency of collisions per mile traveled.'
    },
    'CS (Collision Speed)': {
        'tag': 'custom/collision_speed',
        'desc': 'Vehicle speed at the moment of impact.'
    },
    'ICT (Collision Interval)': {
        'tag': 'custom/collision_interval',
        'desc': 'Time/steps interval of continuous safe driving.'
    },
    'ACD (Avg Center Dev)': {
        'tag': 'custom/avg_center_dev',
        'desc': 'Average deviation from the lane center (m).'
    },
    'Smoothness': {
        'tag': 'custom/mean_steer_smoothness_x100',
        'desc': 'Magnitude of sharp steering variations (steering jitter).'
    },
    'Total Reward': {
        'tag': 'custom/total_reward',
        'desc': 'Cumulative reward including all penalties and bonuses.'
    },
    'GT Reward': {
        'tag': 'rollout/ep_gt_rew_mean',
        'desc': 'Environment Ground Truth reward (raw performance).'
    }
}

def extract_metrics(path, name):
    print(f"Extracting data for {name}...")
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    
    available_tags = ea.Tags()['scalars']
    run_data = {}
    
    for label, info in metrics_info.items():
        tag = info['tag']
        if tag in available_tags:
            scalars = ea.Scalars(tag)
            df = pd.DataFrame([{
                'step': e.step,
                'value': e.value
            } for e in scalars])
            run_data[label] = df
        else:
            print(f"Warning: Tag {tag} not found in {name}")
    
    return run_data

def smooth_data(data, weight=0.8):
    if len(data) == 0: return []
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def generate_dashboard():
    baseline_data = extract_metrics(baseline_path, "Baseline")
    ours_data = extract_metrics(ours_path, "Ours")
    
    # Reorder labels to match user request: 3 per row, Smoothness last
    # Current labels: RC, AS, CR, CPM, CS, ICT, ACD, Smoothness, Total Reward, GT Reward
    ordered_labels = [
        'RC (Routes Completed)', 'AS (Avg Speed)', 'CR (Collision Rate)',
        'CPM (Collisions Per Mile)', 'CS (Collision Speed)', 'ICT (Collision Interval)',
        'ACD (Avg Center Dev)', 'Total Reward', 'GT Reward',
        'Smoothness'
    ]
    
    num_metrics = len(ordered_labels)
    cols = 3
    rows = (num_metrics + cols - 1) // cols
    
    # Custom titles with descriptions
    titles = [f"<b>{label}</b><br><span style='font-size: 10px;'>{metrics_info[label]['desc']}</span>" for label in ordered_labels]
    
    # Define specs for the last row to span all columns
    specs = [[{} for _ in range(cols)] for _ in range(rows - 1)]
    specs.append([{"colspan": cols}, None, None]) # Smoothness spans 3 columns
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.05,
        specs=specs
    )
    
    colors = {'Baseline': '#636EFA', 'Ours': '#00CC96'} 
    
    for i, label in enumerate(ordered_labels):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        # Add Baseline
        if label in baseline_data:
            df = baseline_data[label]
            # Raw
            fig.add_trace(
                go.Scatter(x=df['step'], y=df['value'], mode='lines', 
                           line=dict(color=colors['Baseline'], width=1),
                           opacity=0.2, name=f'Baseline {label} (raw)',
                           legendgroup='Baseline', showlegend=False),
                row=row, col=col
            )
            # Smoothed
            smoothed = smooth_data(df['value'].tolist())
            fig.add_trace(
                go.Scatter(x=df['step'], y=smoothed, mode='lines', 
                           line=dict(color=colors['Baseline'], width=3),
                           name='Baseline (154046)', legendgroup='Baseline',
                           showlegend=(i == 0),
                           hovertemplate='<b>Baseline</b><br>Step: %{x}<br>Value: %{y:.4f}'),
                row=row, col=col
            )
            
        # Add Ours
        if label in ours_data:
            df = ours_data[label]
            # Raw
            fig.add_trace(
                go.Scatter(x=df['step'], y=df['value'], mode='lines', 
                           line=dict(color=colors['Ours'], width=1),
                           opacity=0.2, name=f'Ours {label} (raw)',
                           legendgroup='Ours', showlegend=False),
                row=row, col=col
            )
            # Smoothed
            smoothed = smooth_data(df['value'].tolist())
            fig.add_trace(
                go.Scatter(x=df['step'], y=smoothed, mode='lines', 
                           line=dict(color=colors['Ours'], width=3),
                           name='Ours (082504)', legendgroup='Ours',
                           showlegend=(i == 0),
                           hovertemplate='<b>Ours</b><br>Step: %{x}<br>Value: %{y:.4f}'),
                row=row, col=col
            )
            
    fig.update_layout(
        title_text="VLM-RL Advanced Performance Comparison Dashboard",
        template="plotly_dark",
        height=450 * rows,
        width=1500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(t=250, b=50, l=50, r=50)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Training Steps", gridcolor='#333')
    fig.update_yaxes(gridcolor='#333')

    # Add vertical lines at specific steps
    red_steps = [410000, 820000, 510000, 100000]
    yellow_steps = [470000, 990000]

    for step in red_steps:
        fig.add_vline(x=step, line_width=2, line_dash="dash", line_color="red", opacity=0.7)
    
    for step in yellow_steps:
        fig.add_vline(x=step, line_width=2, line_dash="dash", line_color="yellow", opacity=0.7)

    output_dir = "tensorboard_analysis"
    output_path = os.path.join(output_dir, "vlmrl_comparison_dashboard_final.html")
    
    print(f"Saving dashboard to {output_path}")
    fig.write_html(output_path, include_plotlyjs='cdn')

if __name__ == "__main__":
    generate_dashboard()

if __name__ == "__main__":
    generate_dashboard()
