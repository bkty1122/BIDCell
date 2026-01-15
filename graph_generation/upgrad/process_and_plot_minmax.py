import json
import pandas as pd
import plotly.express as px
import numpy as np
import os

def process_minmax_and_plot():
    # Paths
    csv_path = r"D:\2512-BROCK-CODING\BIDCell\metric_ablated_loss.csv"
    json_path = r"D:\2512-BROCK-CODING\BIDCell\ugrad_ablation_results\ablation_medians.json"
    output_dir = r"D:\2512-BROCK-CODING\BIDCell\graph_generation\upgrad"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("--- Step 1: Processing Data & Calculating Means ---")
    
    # store Processed Means
    method_means = {}
    
    # 1. Process CSV Data
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Mapping CSV Metric Name -> Standard Key
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

    # Group by Method in CSV
    for method in df['Method'].unique():
        method_df = df[df['Method'] == method]
        method_data = {}
        
        for _, row in method_df.iterrows():
            metric_name = row['Metric']
            if metric_name in metric_map_csv:
                # Get mean of the 5 cols
                # Clean strings if necessary
                vals = []
                for col in ablation_cols:
                    val_str = str(row[col])
                    if '%' in val_str:
                        val_str = val_str.replace('%', '')
                        # If percent, maybe divide by 100? Assuming 57.54% -> 0.5754 if range is 0-1? 
                        # But F1 Purity isn't in our target metrics list.
                        # Spatial metrics usually aren't %.
                    vals.append(float(val_str))
                
                std_key = metric_map_csv[metric_name]
                method_data[std_key] = np.mean(vals)
        
        if method_data:
            method_means[method] = method_data

    # 2. Process UGrad JSON
    print(f"Reading JSON: {json_path}")
    with open(json_path, 'r') as f:
        ugrad_json = json.load(f)
    
    ugrad_data = {}
    # metrics list from one of the keys
    first = next(iter(ugrad_json))
    all_metrics = ugrad_json[first].keys()
    
    for metric in all_metrics:
        vals = []
        for col in ablation_cols:
            if col in ugrad_json:
                val = ugrad_json[col].get(metric)
                if val is not None:
                    vals.append(val)
        if vals:
             ugrad_data[metric] = np.mean(vals)
             
    method_means["UGrad"] = ugrad_data

    # 3. Save Summary JSON (Means)
    means_output_path = os.path.join(output_dir, "pooled_means_all_methods.json")
    print(f"Saving Method Means to {means_output_path}")
    def np_encoder(object):
        if isinstance(object, np.generic):
            return object.item()
    with open(means_output_path, 'w') as f:
        json.dump(method_means, f, indent=4, default=np_encoder)

    print("--- Step 2: Normalization (Min-Max Scaling) ---")
    
    # Desired Metrics Order
    ordered_metrics = [
        "total_transcripts", "total_genes", "area", 
        "elongation", "compactness", "sphericity", 
        "solidity", "convexity", "circularity", "density"
    ]
    
    # Calculate Max for each metric across ALL methods
    max_values = {}
    for metric in ordered_metrics:
        vals = []
        for method in method_means:
            val = method_means[method].get(metric)
            if val is not None:
                vals.append(val)
        max_values[metric] = np.max(vals) if vals else 1.0

    # Create Normalized Data (Value / Max)
    # Using 'x / max' ensures values are in [0, 1] relative to the best performer for that metric
    # This is standard for radar plots where the center is 0. 
    # True "Min-Max" (x-min)/(max-min) would put the worst performer at 0 (center point), 
    # which might be visually misleading for non-zero-sum metrics.
    
    plot_labels_map = {
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
    
    plot_labels = [plot_labels_map.get(m, m) for m in ordered_metrics]
    
    # Prepare DataFrame
    data_for_df = {"Metric": plot_labels}
    
    method_order = ["Default Summation", "Aligned-MTL (Median-Scaled)", "Aligned-MTL (Min-Scaled)", "STCH (μ = 0.0005)", "UGrad"]
    
    for method in method_order:
        if method not in method_means: continue
        
        data_for_df[method] = []
        for i, metric in enumerate(ordered_metrics):
            val = method_means[method].get(metric, 0)
            mx = max_values[metric]
            
            norm_val = val / mx if mx != 0 else 0
            data_for_df[method].append(norm_val)
            
    df = pd.DataFrame(data_for_df)
    
    # Save Normalized Data
    norm_output_path = os.path.join(output_dir, "normalized_means_for_plotting.csv")
    df.to_csv(norm_output_path, index=False)
    print(f"Saved normalized data to {norm_output_path}")

    print("--- Step 3: Generating Plot ---")
    
    # Melt
    df_long = df.melt(id_vars="Metric", var_name="Method", value_name="Value")
    
    fig = px.line_polar(df_long, r="Value", theta="Metric", color="Method", line_close=True)
    
    fig.update_traces(fill=None, line_width=2)
    fig.update_layout(
        title={
            'text': "Normalized Spatial Metrics (Mean of Ablations)",
            'y': 0.98,
            'x': 0.02,
            'xanchor': 'left',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.05], # Scaled to [0, 1]
                tickfont=dict(size=14),
            ),
            angularaxis=dict(
                tickfont=dict(size=14),
            )
        ),
        legend=dict(
            title=dict(text="Method"),
            font=dict(size=14)
        ),
        font=dict(size=14)
    )
    
    out_png = os.path.join(output_dir, "fig8_methods_minmax_radar.png")
    out_html = os.path.join(output_dir, "fig8_methods_minmax_radar.html")
    
    print(f"Saving plot to {out_png}")
    try:
        fig.write_image(out_png, width=1200, height=800, scale=2)
        fig.write_html(out_html)
        print("✅ Success!")
    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == "__main__":
    process_minmax_and_plot()
