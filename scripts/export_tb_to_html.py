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
    # Find the event file
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None
    
    # We assume the largest/newest one is the main one if there are multiple, 
    # but usually there's one main one for a run.
    # Let's just pick the first one found or sort by size/time.
    event_file = os.path.join(log_dir, event_files[0])
    print(f"Processing event file: {event_file}")

    # Load the event accumulator
    # size_guidance sets how many events to load. 0 means all.
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    # Extract scalars
    tags = ea.Tags()['scalars']
    data = {}

    for tag in tags:
        scalars = ea.Scalars(tag)
        # We store: step, value, wall_time
        tag_data = []
        for event in scalars:
            tag_data.append({
                'step': event.step,
                'value': event.value,
                'wall_time': event.wall_time
            })
        data[tag] = tag_data
    
    return data

def generate_html_plots(data, output_path):
    """
    Generates a standalone HTML file with Plotly plots for the extracted data.
    """
    if not data:
        print("No data to plot.")
        return

    # Create a list of tags
    tags = list(data.keys())
    if not tags:
        print("No scalar tags found.")
        return

    # Sort tags for better organization (e.g., group by train/, eval/)
    tags.sort()

    # Create figure with a dropdown or subplots? 
    # TensorBoard uses a list of plots. A massive column of plots might be heavy.
    # Let's try to group them or just list them all.
    # A grid of plots is nice.
    
    num_plots = len(tags)
    cols = 2
    rows = (num_plots + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, 
        cols=cols, 
        subplot_titles=tags,
        vertical_spacing=0.05,
        horizontal_spacing=0.1
    )

    for i, tag in enumerate(tags):
        tag_data = data[tag]
        df = pd.DataFrame(tag_data)
        
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        fig.add_trace(
            go.Scatter(
                x=df['step'], 
                y=df['value'], 
                mode='lines', 
                name=tag,
                hovertemplate='Step: %{x}<br>Value: %{y:.4f}<extra></extra>'
            ),
            row=row, 
            col=col
        )
        
        # Add simpler axis labels?
        # fig.update_xaxes(title_text="Step", row=row, col=col)
        # fig.update_yaxes(title_text="Value", row=row, col=col)

    fig.update_layout(
        height=300 * rows, 
        width=1200, 
        title_text="TensorBoard Metrics Export",
        showlegend=False,
        template="plotly_white"
    )

    # Save to HTML
    print(f"Saving HTML to {output_path}")
    fig.write_html(output_path, include_plotlyjs='cdn')

def main():
    if len(sys.argv) < 2:
        print("Usage: python export_tb_to_html.py <log_dir> [output_html_path]")
        sys.exit(1)

    log_dir = sys.argv[1]
    
    # Ensure export directory exists
    export_dir = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard_analysis"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Use the name of the log directory for the filename
        run_name = os.path.basename(os.path.normpath(log_dir))
        output_path = os.path.join(export_dir, f"{run_name}_plots.html")

    data = extract_tensorboard_data(log_dir)
    if data:
        generate_html_plots(data, output_path)
        print("Done.")

if __name__ == "__main__":
    main()
