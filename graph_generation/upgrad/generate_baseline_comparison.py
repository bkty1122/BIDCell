import json
import pandas as pd
import plotly.express as px
import numpy as np
import os

def generate_baseline_radar():
    # Paths
    csv_path = r"D:\2512-BROCK-CODING\BIDCell\data\previous-results-ref\radar-graph-comparasion-raw-data.csv"
    ugrad_json_path = r"D:\2512-BROCK-CODING\BIDCell\data\previous-results-ref\ugrad-lr-07.json"
    output_dir = r"D:\2512-BROCK-CODING\BIDCell\graph_generation\upgrad"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("--- Loading Baseline Data (LR=10^-7) ---")
    
    # 1. Load Reference Methods from CSV
    print(f"Reading CSV: {csv_path}")
    df_csv = pd.read_csv(csv_path)
    
    # Extract data relative to '1.00E-07' column
    # Structure: Metric Category, Method, 1.00E-07
    
    # Map CSV metrics to standard keys
    metric_map_csv_to_std = {
        "Total Transcripts": "total_transcripts",
        "Total Genes": "total_genes",
        "Cell Area": "area",
        "Elongation": "elongation",
        "Compactness": "compactness",
        "Sphericity": "sphericity",
        "Solidity": "solidity",
        "Convexity": "convexity",
        "Circularity": "circularity",
        "Density": "density"
    }

    data_map = {} # Method -> {Metric -> Value}

    for _, row in df_csv.iterrows():
        method = row['Method']
        metric_cat = row['Metric Category']
        val = row['1.00E-07']
        
        if metric_cat in metric_map_csv_to_std:
            std_metric = metric_map_csv_to_std[metric_cat]
            if method not in data_map:
                data_map[method] = {}
            data_map[method][std_metric] = float(val)

    # 2. Load UGrad Data
    print(f"Reading JSON: {ugrad_json_path}")
    with open(ugrad_json_path, 'r') as f:
        ugrad_full = json.load(f)
    
    # UGrad json structure: "LR=1E-07": {metrics...}
    ugrad_vals = ugrad_full.get("LR=1E-07", {})
    
    if not ugrad_vals:
        print("Warning: UGrad data for LR=1E-07 not found!")
    
    data_map["UGrad"] = ugrad_vals
    
    print("--- Normalizing Data (Max-Scaling) ---")
    
    # Metrics to plot
    ordered_metrics = [
        "total_transcripts", "total_genes", "area", 
        "elongation", "compactness", "sphericity", 
        "solidity", "convexity", "circularity", "density"
    ]
    
    # 3. Calculate Max per metric
    max_values = {}
    for m in ordered_metrics:
        vals = []
        for method in data_map:
            v = data_map[method].get(m)
            if v is not None:
                vals.append(v)
        max_values[m] = np.max(vals) if vals else 1.0

    print("Max Values per metric:", max_values)
    
    # 4. Normalize
    normalized_data = {"Metric": []}
    
    # Labels with *
    labels_map = {
        "total_transcripts": "total_transcripts*",
        "total_genes": "total_genes*",
        "area": "cell_area*",
        "elongation": "elongation", # usually no * in ref image? prompt said "Normalized spatial measures"
        "compactness": "compactness*",
        "sphericity": "sphericity", 
        "solidity": "solidity",
        "convexity": "convexity", 
        "circularity": "circularity*",
        "density": "density*"
    }
    # Check reference image for exact labels
    # Image (Fig 4.17): total_transcripts*, density*, circularity*, convexity*, solidity*, sphericity*, compactness*, elongation*, cell_area*, total_genes*
    # ALL have * except maybe elongation? Ah, reference image has elongation* too (Step 129).
    # Wait, Step 129 image shows "elongation*".
    # Step 76 image shows "normalized spatial measures".
    # I will add * to all.
    
    final_labels = [labels_map.get(m, m+"*") if "*" not in labels_map.get(m, "") else labels_map.get(m) for m in ordered_metrics]
    normalized_data["Metric"] = final_labels

    method_list = ["default", "amtl-min", "amtl-median", "stch-mu-0.0005", "UGrad"]
    
    for method in method_list:
        normalized_data[method] = []
        for i, m in enumerate(ordered_metrics):
            val = data_map.get(method, {}).get(m, 0)
            mx = max_values[m]
            norm_val = val / mx if mx != 0 else 0
            normalized_data[method].append(norm_val)

    df_norm = pd.DataFrame(normalized_data)
    
    # Melt
    df_long = df_norm.melt(id_vars="Metric", var_name="Method", value_name="Normalized Value")
    
    # 5. Plot
    fig = px.line_polar(df_long, r="Normalized Value", theta="Metric", color="Method", line_close=True)
    
    fig.update_traces(fill=None, line_width=2)
    fig.update_layout(
        title={
            'text': "Normalized Spatial Measures (Baseline)",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.05],
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
    
    out_png = os.path.join(output_dir, "fig8_baseline_comparison.png")
    out_html = os.path.join(output_dir, "fig8_baseline_comparison.html")
    print(f"Saving to {out_png}")
    
    try:
        fig.write_image(out_png, width=1200, height=800, scale=2)
        fig.write_html(out_html)
        print("✅ Plot generated successfully!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_baseline_radar()
