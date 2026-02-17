import os
import sys
import json
import argparse
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def extract_tensorboard_data(log_dir):
    """
    Extracts scalar data from TensorBoard event files in the given directory.
    """
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None
    
    event_file = os.path.join(log_dir, event_files[0])
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
    Generates a comparison HTML with multiple runs overlayed.
    runs_data: dict {run_name: {tag: df}}
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
    
    fig = make_subplots(
        rows=rows, 
        cols=cols, 
        subplot_titles=tags,
        vertical_spacing=0.02,
        horizontal_spacing=0.05
    )

    # Colors for different runs
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

    fig.update_layout(
        height=400 * rows, 
        width=1600, 
        title_text="TensorBoard Comparison Dashboard",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    print(f"Saving comparison HTML to {output_path}")
    fig.write_html(output_path, include_plotlyjs='cdn')

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_tb_runs.py <log_dir1> <log_dir2> ...")
        sys.exit(1)

    log_dirs = sys.argv[1:]
    runs_data = {}
    
    for log_dir in log_dirs:
        if not os.path.isdir(log_dir):
            print(f"Skipping {log_dir}: Not a directory")
            continue
            
        run_name = os.path.basename(os.path.normpath(log_dir))
        print(f"Extracting data for run: {run_name}")
        data = extract_tensorboard_data(log_dir)
        if data:
            runs_data[run_name] = data

    if runs_data:
        export_dir = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard_analysis"
        os.makedirs(export_dir, exist_ok=True)
        output_path = os.path.join(export_dir, "comparison_dashboard.html")
        generate_comparison_html(runs_data, output_path)
        print("Comparison Done.")

if __name__ == "__main__":
    main()
