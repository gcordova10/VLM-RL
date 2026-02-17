import os
import json
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def extract_tensorboard_data_from_file(event_file):
    """
    Extracts ALL scalar data directly from the given tfevents file.
    """
    if not os.path.exists(event_file):
        print(f"File not found: {event_file}")
        return None
    
    # size_guidance 0 means extract ALL samples
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    tags = ea.Tags()['scalars']
    data = {}
    for tag in tags:
        scalars = ea.Scalars(tag)
        data[tag] = pd.DataFrame([{
            'step': event.step,
            'value': event.value,
            'wall_time': event.wall_time
        } for event in scalars])
    
    return data

def generate_comparison_html(runs_data, output_path):
    """
    Generates a comparison HTML with ALL runs overlayed using the EXACT logic of the original script.
    """
    # Get all unique tags across all runs
    all_tags = set()
    for run_name in runs_data:
        all_tags.update(runs_data[run_name].keys())
    
    tags = sorted(list(all_tags))
    if not tags:
        print("No scalar tags found.")
        return

    num_plots = len(tags)
    cols = 2
    rows = (num_plots + cols - 1) // cols
    
    # Exact same subplot configuration as the original script
    fig = make_subplots(
        rows=rows, 
        cols=cols, 
        subplot_titles=tags,
        vertical_spacing=0.02,
        horizontal_spacing=0.05
    )

    # Colors used in the original script
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for i, tag in enumerate(tags):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        for run_idx, (run_name, run_metrics) in enumerate(runs_data.items()):
            if tag in run_metrics:
                df = run_metrics[tag]
                color = colors[run_idx % len(colors)]
                
                fig.add_trace(
                    go.Scatter(
                        x=df['step'], 
                        y=df['value'], 
                        mode='lines', 
                        name=run_name,
                        legendgroup=run_name,
                        showlegend=(i == 0), # Only show legend once per run
                        line=dict(color=color),
                        hovertemplate=f'Run: {run_name}<br>Step: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>'
                    ),
                    row=row, 
                    col=col
                )

    # Exact same layout configuration as the original script
    fig.update_layout(
        height=400 * rows, 
        width=1600, 
        title_text="TensorBoard Comparison Dashboard (Restored Original View)",
        template="plotly_white", # The original used white template
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    print(f"Saving comparison HTML to {output_path}")
    # Write using the same method as the original (generates the same HTML structure)
    fig.write_html(output_path, include_plotlyjs='cdn')

def main():
    # Exact files provided by the user
    runs_to_compare = {
        "VLM-RL (baseline)": "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0",
        "CLG-Smooth (ours)": "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"
    }
    
    runs_data = {}
    for run_name, file_path in runs_to_compare.items():
        print(f"Extracting ALL data for run: {run_name}")
        data = extract_tensorboard_data_from_file(file_path)
        if data:
            runs_data[run_name] = data

    if runs_data:
        export_dir = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard_analysis"
        output_path = os.path.join(export_dir, "comparison_dashboard_nuevo.html")
        generate_comparison_html(runs_data, output_path)
        print("✅ Dashboard 'comparison_dashboard_nuevo.html' successfully restored to original appearance.")

if __name__ == "__main__":
    main()
