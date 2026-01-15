import json
import pandas as pd
import plotly.express as px
import numpy as np
import os

def generate_aligned_radar():
    # Paths
    csv_path = r"D:\2512-BROCK-CODING\BIDCell\data\previous-results-ref\radar-graph-comparasion-raw-data.csv"
    ugrad_lr07_path = r"D:\2512-BROCK-CODING\BIDCell\data\previous-results-ref\ugrad-lr-07.json"
    ablation_json_path = r"D:\2512-BROCK-CODING\BIDCell\ugrad_ablation_results\ablation_medians.json"
    output_dir = r"D:\2512-BROCK-CODING\BIDCell\graph_generation\upgrad"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("--- 1. Loading Reference Data (CSV) ---")
    df_csv = pd.read_csv(csv_path)
    
    # Map CSV metrics to standard keys
    metric_map = {
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

    # Extract Reference Values for 'default' and other methods
    ref_methods = {}
    for method in df_csv['Method'].unique():
        ref_methods[method] = {}
        m_df = df_csv[df_csv['Method'] == method]
        for _, row in m_df.iterrows():
            cat = row['Metric Category']
            if cat in metric_map:
                key = metric_map[cat]
                ref_methods[method][key] = float(row['1.00E-07'])

    ref_default = ref_methods['default']
    print("Reference Default (1E-07):", ref_default)

    print("--- 2. Loading User Baseline Data (UGrad 1E-07) ---")
    with open(ugrad_lr07_path, 'r') as f:
        user_lr07_json = json.load(f)
    user_default = user_lr07_json.get("LR=1E-07", {})
    print("User Default (1E-07):", user_default)
    
    print("--- 3. Calculating Unit Scaling Factors ---")
    # Identify metrics with unit mismatches (Density, Circularity, Compactness)
    # We will derive a scaling factor: Factor = Ref / User
    # So that User_Scaled = User * Factor ~= Ref
    
    scaling_factors = {}
    # Only scale meaningful unit-mismatch metrics
    metrics_to_scale = ["density", "circularity", "compactness"]
    
    for m in metric_map.values():
        if m in metrics_to_scale:
            if m in user_default and m in ref_default:
                if user_default[m] != 0:
                    factor = ref_default[m] / user_default[m]
                    scaling_factors[m] = factor
                    print(f"  Scaling factor for {m}: {factor:.4e}")
                else:
                    scaling_factors[m] = 1.0
            else:
                scaling_factors[m] = 1.0
        else:
            scaling_factors[m] = 1.0 # No scaling for others
            
    print("--- 4. Loading and Scaling UGrad Ablation Data ---")
    with open(ablation_json_path, 'r') as f:
        ablation_json = json.load(f)
    
    # Calculate Mean of UGrad (across ablations)
    ablation_cols = ['ne', 'os', 'cc', 'mu', 'pn']
    ugrad_means = {}
    
    first = next(iter(ablation_json))
    all_metrics = ablation_json[first].keys()
    
    for metric in all_metrics:
        vals = []
        for col in ablation_cols:
            if col in ablation_json:
                val = ablation_json[col].get(metric)
                if val is not None:
                    # Apply Unit Scaling immediately
                    if metric in scaling_factors:
                        val = val * scaling_factors[metric]
                    vals.append(val)
        if vals:
             ugrad_means[metric] = np.mean(vals)
             
    print("Scaled UGrad Means:", ugrad_means)
    
    # Add UGrad to the methods list
    ref_methods["UGrad"] = ugrad_means

    print("--- 5. Normalizing by Baseline (Ref Default) ---")
    # Reference Graph Strategy: Normalize everything by 'default' (so default=1.0)
    
    ordered_metrics = [
        "total_transcripts", "total_genes", "area", 
        "elongation", "compactness", "sphericity", 
        "solidity", "convexity", "circularity", "density"
    ]
    
    plot_labels_map = {
        "total_transcripts": "total_transcripts*",
        "total_genes": "total_genes*",
        "area": "cell_area*",
        "elongation": "elongation*",
        "compactness": "compactness*",
        "sphericity": "sphericity*", 
        "solidity": "solidity*",
        "convexity": "convexity*",
        "circularity": "circularity*",
        "density": "density*"
    }
    plot_labels = [plot_labels_map.get(m, m) for m in ordered_metrics]
    
    final_data = {"Metric": plot_labels}
    method_order = ["default", "amtl-min", "amtl-median", "stch-mu-0.0005", "UGrad"]
    
    for method in method_order:
        if method not in ref_methods: continue
        
        final_data[method] = []
        for m in ordered_metrics:
            val = ref_methods[method].get(m, 0)
            baseline = ref_default.get(m, 1.0)
            
            if baseline == 0: baseline = 1.0
            
            norm_val = val / baseline
            final_data[method].append(norm_val)

    df_plot = pd.DataFrame(final_data)
    
    print("--- 6. Generate Plot ---")
    df_long = df_plot.melt(id_vars="Metric", var_name="Method", value_name="Normalized Value")
    
    fig = px.line_polar(df_long, r="Normalized Value", theta="Metric", color="Method", line_close=True)
    
    fig.update_traces(fill=None, line_width=2)
    fig.update_layout(
        title={
            'text': "Normalized Spatial Metrics (Baseline Corrected)",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 2.0], # Similar to ref image range (approx 0-2)
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
    
    out_png = os.path.join(output_dir, "fig8_aligned_comparison.png")
    out_html = os.path.join(output_dir, "fig8_aligned_comparison.html")
    
    print(f"Saving to {out_png}")
    try:
        fig.write_image(out_png, width=1200, height=800, scale=2)
        fig.write_html(out_html)
        print("✅ Plot generated successfully!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_aligned_radar()
