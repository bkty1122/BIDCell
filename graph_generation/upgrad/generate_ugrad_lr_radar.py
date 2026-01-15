import json
import pandas as pd
import plotly.express as px
import numpy as np
import os

def generate_ugrad_lr_radar_plot():
    # Path to the JSON data
    json_path = r"D:\2512-BROCK-CODING\BIDCell\ugrad_results\ugrad_lr_medians.json"
    
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"Error: Data file not found at {json_path}")
        print("Please ensure you have 'ugrad_lr_medians.json' with learning rate data.")
        return

    # Load the data
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data_values = json.load(f)

    # Define metrics mapping based on the provided image
    # The '*' indicates normalization
    labels_map = {
        "total_transcripts": "total_transcripts*",
        "total_genes": "total_genes*",
        "area": "cell_area*",  # Mapped from 'area' to 'cell_area*'
        "elongation": "elongation",
        "compactness": "compactness*",
        "sphericity": "sphericity",
        "solidity": "solidity",
        "convexity": "convexity",
        "circularity": "circularity*",
        "density": "density*"
    }
    
    # Order of metrics in the graph (clockwise starting from top)
    # Based on the reference image: total_transcripts, total_genes, cell_area, elongation, compactness, sphericity, solidity, convexity, circularity, density
    ordered_metrics = [
        "total_transcripts", "total_genes", "area", 
        "elongation", "compactness", "sphericity", 
        "solidity", "convexity", "circularity", "density"
    ]
    
    # Ensure all metrics exist in the data
    first_key = next(iter(data_values))
    available_keys = data_values[first_key].keys()
    
    # Filter ordered_metrics to only those available in data (robustness)
    ordered_metrics = [m for m in ordered_metrics if m in available_keys]
    
    plot_labels = [labels_map.get(m, m) for m in ordered_metrics]
    
    # Calculate max values for normalization
    col_maxs = {}
    for col in ordered_metrics:
        # Extract all values for this column across all learning rates
        values = [data_values[k].get(col, 0) for k in data_values.keys()]
        col_maxs[col] = np.max(values)

    # Prepare data for Plotly
    plot_data = {"Metric": plot_labels}
    experiment_keys = list(data_values.keys()) # Expected: LR=1E-05, LR=1E-06, etc.
    
    # Sort keys by learning rate value (largest to smallest)
    try:
        # Extract number after '=' and convert to float
        experiment_keys.sort(key=lambda x: float(x.split('=')[1]) if '=' in x else x, reverse=True) 
    except Exception as e:
        print(f"Sorting keys alphabetically due to: {e}")
        experiment_keys.sort()

    for key in experiment_keys:
        plot_data[key] = []
        for metric in ordered_metrics:
            label = labels_map.get(metric, metric)
            value = data_values[key].get(metric, 0)
            
            if "*" in label:
                # Normalize
                max_val = col_maxs[metric]
                normalized_val = value / max_val if max_val != 0 else 0
                plot_data[key].append(normalized_val)
            else:
                plot_data[key].append(value)

    df = pd.DataFrame(plot_data)
    
    # Melt the DataFrame for plotly
    key_name = "Learning Rate"
    df_long = df.melt(id_vars="Metric", var_name=key_name, value_name="Value")

    # Generate the plot
    fig = px.line_polar(df_long, r="Value", theta="Metric", color=key_name, line_close=True)
    
    # Styling
    fig.update_traces(fill=None)
    fig.update_layout(
        title={
            'text': "UGrad",
            'y': 0.98,  # Position near top
            'x': 0.02,  # Position at left corner
            'xanchor': 'left',
            'yanchor': 'top',
            'font': dict(size=36)  # Larger title
        },
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1], # Normalized scale 0-1
                tickfont=dict(size=24),  # Radial axis tick labels
            ),
            angularaxis=dict(
                tickfont=dict(size=24),  # Angular axis tick labels (metric names)
            )
        ),
        legend=dict(
            title=dict(text="Learning Rate"),
            font=dict(size=24)  # Legend items
        ),
        font=dict(size=18)  # Fallback/default font size
    )
    
    output_dir = r"D:\2512-BROCK-CODING\BIDCell\graph_generation\upgrad"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_html = os.path.join(output_dir, "ugrad_lr_radar_plot.html")
    print(f"Saving plot to {output_html}")
    fig.write_html(output_html)

    output_png = os.path.join(output_dir, "ugrad_lr_radar_plot.png")
    print(f"Saving plot to {output_png}")
    try:
        # Increasing scale for higher resolution
        fig.write_image(output_png, width=1200, height=800, scale=2)
        print("✅ Plot generated successfully!")
    except Exception as e:
        print(f"Error saving PNG: {e}")
        print("Note: PNG export requires 'kaleido'. You can install it with: pip install kaleido")

if __name__ == "__main__":
    generate_ugrad_lr_radar_plot()
