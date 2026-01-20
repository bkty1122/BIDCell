import json
import pandas as pd
import plotly.express as px
import numpy as np
import os

def generate_normalized_comparison_radar():
    """
    Generate a radar plot comparing multiple methods normalized by the 'default' method.
    The 'default' method will be 1.0 for all metrics.
    """
    # Paths to data files
    csv_path = r"D:\2512-BROCK-CODING\BIDCell\data\previous-results-ref\radar-graph-comparasion-raw-data.csv"
    ugrad_json_path = r"D:\2512-BROCK-CODING\BIDCell\data\previous-results-ref\ugrad.json"
    
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    if not os.path.exists(ugrad_json_path):
        print(f"Error: JSON file not found at {ugrad_json_path}")
        return
    
    print(f"Loading data from {csv_path}...")
    # Read CSV data
    df_csv = pd.read_csv(csv_path)
    
    print(f"Loading data from {ugrad_json_path}...")
    # Read UGrad JSON data
    with open(ugrad_json_path, 'r') as f:
        ugrad_data = json.load(f)
    
    # --- Configuration for UGrad Data Selection ---
    # Choose the key from ugrad.json (e.g., "LR=1E-05", "LR=1E-07")
    ugrad_key = "LR=1E-09" 
    # Choose the legend label (e.g., "ugrad-lr-05", "ugrad-lr-07")
    ugrad_label = "ugrad-lr-09"
    
    if ugrad_key not in ugrad_data:
        print(f"Error: Key {ugrad_key} not found in {ugrad_json_path}")
        return

    # Extract selected UGrad data
    ugrad_selected_data = ugrad_data[ugrad_key]
    
    # Metric mapping from JSON keys to CSV metric categories
    metric_mapping = {
        "area": "Cell Area",
        "circularity": "Circularity",
        "compactness": "Compactness",
        "convexity": "Convexity",
        "density": "Density",
        "elongation": "Elongation",
        "solidity": "Solidity",
        "sphericity": "Sphericity",
        "total_genes": "Total Genes",
        "total_transcripts": "Total Transcripts"
    }
    
    # Create a dictionary to store all method data
    methods_data = {}
    
    # Extract data from CSV for each method
    for method in df_csv['Method'].unique():
        methods_data[method] = {}
        method_df = df_csv[df_csv['Method'] == method]
        for _, row in method_df.iterrows():
            metric = row['Metric Category']
            value = row['1.00E-07']
            methods_data[method][metric] = value
    
    # Add UGrad data
    methods_data[ugrad_label] = {}
    for json_key, csv_metric in metric_mapping.items():
        if json_key in ugrad_selected_data:
            methods_data[ugrad_label][csv_metric] = ugrad_selected_data[json_key]
    
    # Get the default method values for normalization
    default_values = methods_data['default']
    
    # Define the order of metrics (matching the reference image)
    ordered_metrics = [
        "Total Transcripts",
        "Total Genes",
        "Cell Area",
        "Elongation",
        "Compactness",
        "Sphericity",
        "Solidity",
        "Convexity",
        "Circularity",
        "Density"
    ]
    
    # Create labels with * to indicate normalization
    plot_labels = [
        "total_transcripts*",
        "total_genes*",
        "cell_area*",
        "elongation*",
        "compactness*",
        "sphericity*",
        "solidity*",
        "convexity*",
        "circularity*",
        "density*"
    ]
    
    # Normalize all methods by default
    normalized_data = {"Metric": plot_labels}
    
    # Order of methods for consistent legend
    method_order = ['default', 'amtl-min', 'amtl-median', 'stch-mu-0.0005', ugrad_label]
    
    for method in method_order:
        if method not in methods_data:
            continue
        normalized_data[method] = []
        for metric in ordered_metrics:
            if metric in methods_data[method] and metric in default_values:
                # Normalize by default value
                normalized_value = methods_data[method][metric] / default_values[metric]
                normalized_data[method].append(normalized_value)
            else:
                normalized_data[method].append(0)
    
    # Create DataFrame
    df = pd.DataFrame(normalized_data)
    
    # Melt for plotly
    df_long = df.melt(id_vars="Metric", var_name="method", value_name="Value")
    
    # Generate the plot
    fig = px.line_polar(df_long, r="Value", theta="Metric", color="method", line_close=True)
    
    # Styling to match reference
    fig.update_traces(fill=None)
    fig.update_layout(
        title={
            'text': "Normalized Spatial Metrics by Method",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 2],  # Adjusted range to accommodate values relative to default (1.0)
                tickfont=dict(size=18),
            ),
            angularaxis=dict(
                tickfont=dict(size=18),
            )
        ),
        legend=dict(
            title=dict(text="method"),
            font=dict(size=16)
        ),
        font=dict(size=14)
    )
    
    # Save outputs
    output_dir = r"D:\2512-BROCK-CODING\BIDCell\graph_generation\upgrad"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_html = os.path.join(output_dir, "normalized_comparison_radar.html")
    print(f"Saving plot to {output_html}")
    fig.write_html(output_html)
    
    output_png = os.path.join(output_dir, "normalized_comparison_radar" + ugrad_label + ".png")
    print(f"Saving plot to {output_png}")
    try:
        fig.write_image(output_png, width=1200, height=800, scale=2)
        print("✅ Plot generated successfully!")
    except Exception as e:
        print(f"Error saving PNG: {e}")
        print("Note: PNG export requires 'kaleido'. You can install it with: pip install kaleido")

if __name__ == "__main__":
    generate_normalized_comparison_radar()
