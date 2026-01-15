import json
import pandas as pd
import plotly.express as px
import numpy as np
import os

def process_and_plot_fig8():
    # Paths
    csv_path = r"D:\2512-BROCK-CODING\BIDCell\metric_ablated_loss.csv"
    json_path = r"D:\2512-BROCK-CODING\BIDCell\ugrad_ablation_results\ablation_medians.json"
    output_dir = r"D:\2512-BROCK-CODING\BIDCell\graph_generation\upgrad"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Process CSV Data
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # metrics of interest for the radar plot
    # Map CSV metric names to a standard key (we'll use the JSON keys as standard)
    # CSV Metric Name -> Standard Key
    metric_map_csv = {
        "total transcripts": "total_transcripts",
        "total genes": "total_genes",
        "cell area": "area",
        "elongation": "elongation",
        "compactness": "compactness",
        "sphericity": "sphericity",
        "solidity": "solidity",
        "convexity": "convexity",
        "circularity": "circularity",
        "density": "density"
    }

    # Columns to average
    ablation_cols = ['ne', 'os', 'cc', 'mu', 'pn']
    
    # Container for aggregated data: {MethodName: {Metric: MeanValue}}
    aggregated_data = {}

    # Cleaning helpers
    def clean_val(x):
        if isinstance(x, str):
            x = x.replace('%', '')
        return float(x)

    # Iterate over unique methods in CSV
    for method in df['Method'].unique():
        method_df = df[df['Method'] == method]
        method_means = {}
        
        for _, row in method_df.iterrows():
            metric_name = row['Metric']
            if metric_name in metric_map_csv:
                # Calculate mean of the 5 columns
                values = [clean_val(row[col]) for col in ablation_cols]
                mean_val = np.mean(values)
                std_key = metric_map_csv[metric_name]
                method_means[std_key] = mean_val
        
        if method_means:
            aggregated_data[method] = method_means

    # 2. Process JSON Data (UGrad)
    print(f"Reading JSON: {json_path}")
    with open(json_path, 'r') as f:
        ablation_json = json.load(f)
    
    # Calculate mean across the 5 ablation keys for UGrad
    # keys in json are "ne", "os", ...
    # structure: "ne": {"area": 123, ...}
    
    ugrad_means = {}
    # Get list of metrics from the first entry
    first_key = next(iter(ablation_json))
    metrics_list = ablation_json[first_key].keys()
    
    for metric in metrics_list:
        # standard key is already matching JSON keys usually
        values = []
        for abl_key in ablation_cols: # ne, os, cc, mu, pn
            if abl_key in ablation_json:
                val = ablation_json[abl_key].get(metric)
                if val is not None:
                    values.append(val)
        
        if values:
            ugrad_means[metric] = np.mean(values)
            
    # Add UGrad to aggregated data
    # Giving it a distinct name for the plot
    aggregated_data["UGrad"] = ugrad_means

    # 3. Save Summary JSON
    summary_json_path = os.path.join(output_dir, "fig8_methods_means_summary.json")
    print(f"Saving summary JSON to {summary_json_path}")
    # Convert numpy types to python types for json serialization
    def convert(o):
        if isinstance(o, np.generic): return o.item()
        raise TypeError
    
    with open(summary_json_path, 'w') as f:
        json.dump(aggregated_data, f, indent=4, default=convert)

    # 4. Prepare Data for Plotting (Normalization)
    # Baseline: "Default Summation"
    baseline_method = "Default Summation"
    
    if baseline_method not in aggregated_data:
        print(f"Error: Baseline method '{baseline_method}' not found in data.")
        return

    baseline_vals = aggregated_data[baseline_method]
    
    # Plot Metrics Order
    plot_metrics_order_keys = [
        "total_transcripts", "total_genes", "area", 
        "elongation", "compactness", "sphericity", 
        "solidity", "convexity", "circularity", "density"
    ]
    
    # Labels for the plot (adding *)
    plot_labels_map = {
        "total_transcripts": "total_transcripts*",
        "total_genes": "total_genes*",
        "area": "cell_area*",
        "elongation": "elongation",
        "compactness": "compactness*",
        "sphericity": "sphericity", # usually not normalized in name but curve looks normalized? 
                                    # In previous requests: density*, cell_area*, etc. 
                                    # Let's stick to the previous request's label convention if possible.
                                    # Re-using the latest map from user prompt or similar
        "solidity": "solidity",
        "convexity": "convexity",
        "circularity": "circularity*",
        "density": "density*"
    }
    # Wait, in the reference "norm-spatial-by-method.png", almost everything has * if normalized? 
    # Usually elongation, solidity, convexity, sphericity are 0-1 metrics already, so maybe not normalized?
    # But the prompt says "normalized spatial measures", and "Default" should be 1.0. 
    # If we normalize everything by Default, EVERYTHING becomes relative (centered at 1.0).
    # So we should apply normalization to ALL desired metrics.
    
    # Let's standardize labels to [Metric]*
    plot_labels = [plot_labels_map.get(k, k+"*") for k in plot_metrics_order_keys]

    # Build DataFrame for Plotly
    final_plot_data = {"Metric": plot_labels}
    
    methods_to_plot = list(aggregated_data.keys())
    # Ensure Default is first or handled? Order doesn't matter much for plotly, but Legend order might.
    # User wanted: Default, Aligned-MTL..., STCH..., UGrad.
    
    for method in methods_to_plot:
        final_plot_data[method] = []
        for k in plot_metrics_order_keys:
            val = aggregated_data[method].get(k, 0)
            base = baseline_vals.get(k, 1) # avoid div by zero if missing
            if base == 0: base = 1 # safety
            
            norm_val = val / base
            final_plot_data[method].append(norm_val)

    df_plot = pd.DataFrame(final_plot_data)
    
    # Melt
    df_long = df_plot.melt(id_vars="Metric", var_name="Method", value_name="Normalized Value")

    # 5. Generate Radar Plot
    fig = px.line_polar(df_long, r="Normalized Value", theta="Metric", color="Method", line_close=True)
    
    fig.update_traces(fill=None, line_width=2)
    fig.update_layout(
        title={
            'text': "Normalized Spatial Measures by Method",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        polar=dict(
            radialaxis=dict(
                visible=True,
                # The data is normalized to 1.0. Range should probably cover the variation.
                # Auto-range is usually okay, but typically [0, 2] or similar is good for comparison.
                tickfont=dict(size=18),
            ),
            angularaxis=dict(
                tickfont=dict(size=18),
            )
        ),
        legend=dict(
            title=dict(text="Method"),
            font=dict(size=16)
        ),
        font=dict(size=14)
    )

    output_png = os.path.join(output_dir, "fig8_normalized_spatial_methods.png")
    output_html = os.path.join(output_dir, "fig8_normalized_spatial_methods.html")
    
    print(f"Saving plot to {output_png}")
    try:
        fig.write_image(output_png, width=1200, height=800, scale=2)
        fig.write_html(output_html)
        print("✅ Plot generated successfully!")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    process_and_plot_fig8()
