
import json
import pandas as pd
import plotly.express as px
import numpy as np
import os

def generate_ablation_radar():
    # Paths
    root_dir = r"d:\2512-BROCK-CODING\BIDCell"
    input_path = os.path.join(root_dir, r"ugrad_ablation_results\ablation_medians.json")
    output_dir = os.path.join(root_dir, r"graph_generation\ablation")
    
    if not os.path.exists(input_path):
        print(f"Data file not found at {input_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    with open(input_path, 'r') as f:
        data_values = json.load(f)
        
    # Define metrics mapping based on the provided reference style
    # The '*' indicates normalization
    labels_map = {
        "total_transcripts": "total_transcripts*",
        "total_genes": "total_genes*",
        "area": "cell_area*",
        "elongation": "elongation",
        "compactness": "compactness*",
        "sphericity": "sphericity",
        "solidity": "solidity",
        "convexity": "convexity",
        "circularity": "circularity*",
        "density": "density*"
    }
    
    # Order of metrics in the graph
    ordered_metrics = [
        "density", "total_transcripts", "total_genes", "area", 
        "elongation", "compactness", "sphericity", 
        "solidity", "convexity", "circularity"
    ]
    
    # Ensure all metrics exist in the data (check first entry)
    # data_values is { "ne": {...}, "os": {...} }
    if not data_values:
        print("No data found in JSON.")
        return

    first_key = next(iter(data_values))
    sample_data = data_values[first_key]
    
    available_metrics = []
    for m in ordered_metrics:
        # Check if metric m is in data
        if m in sample_data:
            available_metrics.append(m)
    
    plot_labels = [labels_map.get(m, m) for m in available_metrics]
    
    # Calculate max values for normalization across all experiments
    col_maxs = {}
    for col in available_metrics:
        values = []
        for key in data_values:
            val = data_values[key].get(col, 0)
            values.append(val)
        col_maxs[col] = np.max(values) if values else 1.0

    # Prepare data for Plotly
    plot_data = {"Metric": plot_labels}
    experiment_keys = list(data_values.keys())
    experiment_keys.sort() # Alphabetical sort for NE, OS, etc.

    for key in experiment_keys:
        plot_data[key] = []
        for metric in available_metrics:
            label = labels_map.get(metric, metric)
            value = data_values[key].get(metric, 0)
            
            # Identify if we should normalize based on label having '*'
            if "*" in label:
                max_val = col_maxs[metric]
                normalized_val = value / max_val if max_val != 0 else 0
                plot_data[key].append(normalized_val)
            else:
                plot_data[key].append(value)

    df = pd.DataFrame(plot_data)
    
    # Melt for plotly
    key_name = "Experiment"
    df_long = df.melt(id_vars="Metric", var_name=key_name, value_name="Value")

    # Generate the plot
    fig = px.line_polar(df_long, r="Value", theta="Metric", color=key_name, line_close=True)
    
    # Styling to match reference
    fig.update_traces(fill=None)
    fig.update_layout(
        title={
            'text': "Ablation Study Comparison",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=12),
            ),
            angularaxis=dict(
                tickfont=dict(size=14),
            )
        ),
        legend=dict(
            title=dict(text="Conditions"),
            font=dict(size=14)
        ),
        font=dict(size=14)
    )
    
    # Save output
    output_html = os.path.join(output_dir, "ablation_radar_plot.html")
    print(f"Saving plot to {output_html}")
    fig.write_html(output_html)

    output_png = os.path.join(output_dir, "ablation_radar_plot.png")
    print(f"Saving plot to {output_png}")
    try:
        fig.write_image(output_png, width=1200, height=800, scale=2)
    except Exception as e:
        print(f"Error saving PNG: {e}")
        print("Note: PNG export requires 'kaleido'. pip install kaleido")

if __name__ == "__main__":
    generate_ablation_radar()
