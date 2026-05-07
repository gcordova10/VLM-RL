import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_data():
    with open("data/radar_metrics_ours.json", "r") as f:
        radar_raw = json.load(f)
    
    with open("driving_quality_analysis/driving_quality_baseline.json", "r") as f:
        dq_baseline = json.load(f)
        
    with open("driving_quality_analysis/driving_quality_ours.json", "r") as f:
        dq_ours = json.load(f)
        
    return radar_raw, dq_baseline, dq_ours

def get_smoothness(dq_data, step):
    step_data = [d['jitter'] for d in dq_data if d['steps'] == step]
    if not step_data: return 1.0 # default
    return sum(step_data) / len(step_data)

def generate_report():
    radar_raw, dq_baseline, dq_ours = load_data()
    
    # Map steps
    bt = radar_raw['baseline']
    ot = radar_raw['ours']
    
    # Helper to build metrics dict
    def get_metrics(run_data, dq_data, step):
        s = str(step)
        m = run_data.get(s, {})
        sm = get_smoothness(dq_data, step)
        return {
            'rc': m.get('routes_completed', 0),
            'cpm': m.get('CPM', 1e6),
            'acd': m.get('avg_center_dev', 1e6),
            'sm': sm,
            'ci': m.get('collision_interval', 0),
            'as': m.get('avg_speed', 0),
            'cs': m.get('collision_speed', 1e6),
            'cr': m.get('collision_rate', 1e6),
            'tr': m.get('total_reward', 0),
            'gr': m.get('ep_gt_rew_mean', 0)
        }

    # Data structure for report (Criteria and their metrics)
    # steps format: (Baseline Step, Ours Step)
    criteria_setup = {
        '1. Balanced (General)': {
            'metrics': ['Routes Completed', 'CPM (inv)', 'Avg Center Dev (inv)', 'Smoothness (inv)'],
            'steps': (470000, 410000),
            'keys': ['rc', 'cpm_inv', 'acd_inv', 'sm_inv']
        },
        '2. Fluid Driving (State)': {
            'metrics': ['Routes Completed', 'Avg Center Dev (inv)', 'Coll Interval'],
            'steps': (990000, 410000),
            'keys': ['rc', 'acd_inv', 'ci']
        },
        '3. High Speed (Efficacy)': {
            'metrics': ['Routes Completed', 'Avg Speed', 'CPM (inv)', 'Coll Speed (inv)'],
            'steps': (470000, 820000),
            'keys': ['rc', 'as', 'cpm_inv', 'cs_inv']
        },
        '4. Learning Consistency': {
            'metrics': ['Coll Interval', 'GT Reward'],
            'steps': (990000, 510000),
            'keys': ['ci', 'gr']
        },
        '5. Technical Fidelity': {
            'metrics': ['Collision Rate (inv)', 'Routes Completed', 'Total Reward', 'Smoothness (inv)', 'Avg Center Dev (inv)'],
            'steps': (470000, 100000),
            'keys': ['cr_inv', 'rc', 'tr', 'sm_inv', 'acd_inv']
        }
    }

    processed_data = {}
    for crit, info in criteria_setup.items():
        b_step, o_step = info['steps']
        b_m = get_metrics(bt, dq_baseline, b_step)
        o_m = get_metrics(ot, dq_ours, o_step)
        
        # Add inversions
        def add_inv(m):
            m['cpm_inv'] = 1.0 / (m['cpm'] + 1e-6)
            m['acd_inv'] = 1.0 / (m['acd'] + 1e-6)
            m['sm_inv'] = 1.0 / (m['sm'] + 1e-6)
            m['cs_inv'] = 1.0 / (m['cs'] + 1e-6)
            m['cr_inv'] = 1.0 / (m['cr'] + 1e-6)
            return m
            
        b_m = add_inv(b_m)
        o_m = add_inv(o_m)
        
        baseline_vals = [b_m[k] for k in info['keys']]
        ours_vals = [o_m[k] for k in info['keys']]
        
        # Normalize 0-100
        norm_b = []
        norm_o = []
        for bv, ov in zip(baseline_vals, ours_vals):
            mx = max(abs(bv), abs(ov), 1e-6)
            norm_b.append((bv / mx) * 100)
            norm_o.append((ov / mx) * 100)
            
        processed_data[crit] = {
            'baseline': norm_b,
            'ours': norm_o,
            'metrics': info['metrics'],
            'steps': (b_step, o_step),
            'raw': {'baseline': baseline_vals, 'ours': ours_vals}
        }

    # Generate the Plots
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}],
               [{'type': 'polar'}, {'type': 'polar'}, None]],
        subplot_titles=[f"<b>{k}</b>" for k in criteria_setup.keys()],
        horizontal_spacing=0.15,
        vertical_spacing=0.2
    )

    colors = {'Baseline': '#636EFA', 'Ours': '#00CC96'}
    
    for i, (crit, data) in enumerate(processed_data.items()):
        row = (i // 3) + 1
        col = (i % 3) + 1
        b_step, o_step = data['steps']
        
        fig.add_trace(go.Scatterpolar(
            r=data['baseline'], theta=data['metrics'],
            fill='toself', name=f'Baseline (Step {b_step//1000}k)',
            line_color=colors['Baseline'], legendgroup='Baseline', showlegend=(i == 0)
        ), row=row, col=col)
        
        fig.add_trace(go.Scatterpolar(
            r=data['ours'], theta=data['metrics'],
            fill='toself', name=f'Ours (Step {o_step//1000}k)',
            line_color=colors['Ours'], legendgroup='Ours', showlegend=(i == 0)
        ), row=row, col=col)

    fig.update_layout(
        template="plotly_dark", height=1000, width=1400,
        margin=dict(t=150, b=100, l=50, r=50),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        title=dict(text="VLM-RL MULTI-CRITERION COMPARISON (LATEST DATA)", font=dict(size=24), x=0.5)
    )

    output_path = "tensorboard_analysis/vlmrl_comparison_dashboard_ours.html"
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"Final Dashboard Generated: {output_path}")

if __name__ == "__main__":
    generate_report()
