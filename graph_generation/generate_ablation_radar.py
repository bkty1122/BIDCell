import json
import pandas as pd
import plotly.express as px
import numpy as np
import os

def generate_ablation_radar_plot():
    # Path to the JSON data
    json_path = r"D:\2512-BROCK-CODING\BIDCell\ugrad_ablation_results\ablation_medians.json"
    
    # Load the data
    with open(json_path, 'r') as f:
        data_values = json.load(f)

    # Define metrics (cols) and labels with normalization markers set by user preferences in snippet
    # Mapping JSON keys to what might be expected or just using them directly.
    # JSON keys: total_transcripts, total_genes, area, elongation, compactness, sphericity, solidity, convexity, circularity, density
    
    # Labels with '*' indicate normalization (dividing by max)
    # Based on user snippet reference:
    # Normalized: total_transcripts, total_genes, area (cell_area), compactness, circularity, density
    # Not normalized: elongation, sphericity, solidity, convexity
    
    labels_map = {
        "total_transcripts": "total_transcripts*",
        "total_genes": "total_genes*",
        "area": "area*",
        "elongation": "elongation",
        "compactness": "compactness*",
        "sphericity": "sphericity",
        "solidity": "solidity",
        "convexity": "convexity",
        "circularity": "circularity*",
        "density": "density*"
    }
    
    # List of metrics to plot
    cols = list(data_values[next(iter(data_values))].keys())
    
    # Calculate max values for each column for normalization
    col_maxs = {}
    for col in cols:
        values = [data_values[key][col] for key in data_values.keys()]
        col_maxs[col] = np.max(values)

    # Prepare data for DataFrame
    # Structure: {"Metric": [label1, label2, ...], "ne": [val1, val2...], "os": [...], ...}
    
    # We want the labels in the plot to match the list in the snippet if possible, or at least be clean.
    # The snippet uses labels list to drive the order.
    
    ordered_metrics = [
        "total_transcripts", "total_genes", "area", 
        "elongation", "compactness", "sphericity", 
        "solidity", "convexity", "circularity", "density"
    ]
    
    plot_labels = [labels_map[m] for m in ordered_metrics]
    
    plot_data = {"Metric": plot_labels}
    
    experiment_keys = list(data_values.keys()) # ne, os, cc, mu, pn
    
    for key in experiment_keys:
        plot_data[key] = []
        for metric in ordered_metrics:
            label = labels_map[metric]
            value = data_values[key][metric]
            
            if "*" in label:
                # Normalize
                normalized_val = value / col_maxs[metric] if col_maxs[metric] != 0 else 0
                plot_data[key].append(normalized_val)
            else:
                plot_data[key].append(value)

    df = pd.DataFrame(plot_data)
    
    # Melt the DataFrame for plotly
    # id_vars = "Metric", var_name = "Ablation Type", value_name = "Value"
    key_name = "Ablation Type"
    df_long = df.melt(id_vars="Metric", var_name=key_name, value_name="Value")

    # Generate the plot
    fig = px.line_polar(df_long, r="Value", theta="Metric", color=key_name, line_close=True)
    
    # Styling as requested
    fig.update_traces(fill=None)
    fig.update_layout(
        title={
            'text': "Ablation Study: Metrics Comparison",
            'font': dict(size=36)
        },
        polar=dict(
            radialaxis=dict(
                visible=True,
                # range=[0, 1], # Range depends on if all are normalized. 
                # Since some are NOT normalized (like solidity ~0.8), and normalized ones are 0-1.
                # If mixed, range 0-1 might cut off data > 1?
                # However, user snippet sets range=[0, 1].
                # Let's check the non-normalized values in JSON:
                # elongation ~0.65, sphericity ~0.66, solidity ~0.75, convexity ~0.86.
                # All seem to be < 1. So range [0, 1] is safe and appropriate.
                range=[0, 1],
                tickfont=dict(size=24),
            ),
            angularaxis=dict(
                tickfont=dict(size=24),
            )
        ),
        legend=dict(font=dict(size=24)),
        font=dict(size=18)
    )
    
    # Show or Save? The user asked to "create a script ... that can transformating hte ablation medians information to the graphs that i want".
    # Usually `fig.show()` opens a browser. I should probably also save it to an HTML or PNG file.
    
    output_dir = r"D:\2512-BROCK-CODING\BIDCell\graph_generation"
    output_html = os.path.join(output_dir, "ablation_radar_plot.html")
    print(f"Saving plot to {output_html}")
    fig.write_html(output_html)

    # Save as PNG
    output_png = os.path.join(output_dir, "ablation_radar_plot.png")
    print(f"Saving plot to {output_png}")
    try:
        fig.write_image(output_png, width=1200, height=800, scale=2)
    except Exception as e:
        print(f"Error saving PNG: {e}")
        print("Ensure 'kaleido' is installed: pip install kaleido")
    
if __name__ == "__main__":
    generate_ablation_radar_plot()
