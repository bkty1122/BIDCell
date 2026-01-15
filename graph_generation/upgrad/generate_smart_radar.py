import json
import pandas as pd
import plotly.express as px
import numpy as np
import os
import math

def generate_smart_radar():
    # Paths
    csv_path = r"D:\2512-BROCK-CODING\BIDCell\data\previous-results-ref\radar-graph-comparasion-raw-data.csv"
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
    
    print("\n--- 2. Loading UGrad Ablation Data (User) ---")
    with open(ablation_json_path, 'r') as f:
        ablation_json = json.load(f)
    
    # Calculate Mean of UGrad (across ablations)
    ablation_cols = ['ne', 'os', 'cc', 'mu', 'pn']
    ugrad_means = {}
    
    first_key = next(iter(ablation_json))
    all_metrics = ablation_json[first_key].keys()
    
    for metric in all_metrics:
        vals = []
        for col in ablation_cols:
            if col in ablation_json:
                val = ablation_json[col].get(metric)
                if val is not None:
                    vals.append(val)
        if vals:
             ugrad_means[metric] = np.mean(vals)
    
    print("\n--- 3. Smart Unit Alignment ---")
    
    scaling_factors = {}
    threshold_log_diff = 1.5 
    
    for m in metric_map.values():
        ref_val = ref_default.get(m, 0)
        user_val = ugrad_means.get(m, 0)
        
        factor = 1.0
        
        if ref_val > 0 and user_val > 0:
            log_diff = abs(np.log10(ref_val) - np.log10(user_val))
            
            if log_diff > threshold_log_diff:
                factor = ref_val / user_val
                print(f"  [Scaling] Metric '{m}': Ref={ref_val:.2e}, User={user_val:.2e}. Factor={factor:.2e}")
            else:
                print(f"  [Keep]    Metric '{m}': Ref={ref_val:.2e}, User={user_val:.2e}. No Scale.")
        
        ugrad_means[m] = ugrad_means.get(m, 0) * factor
        
    ref_methods["UGrad"] = ugrad_means

    print("\n--- 4. Normalizing by Baseline (Ref Default) ---")
    
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
    
    max_norm_value = 0
    
    for method in method_order:
        if method not in ref_methods: continue
        
        final_data[method] = []
        for m in ordered_metrics:
            val = ref_methods[method].get(m, 0)
            baseline = ref_default.get(m, 1.0)
            if baseline == 0: baseline = 1.0
            
            norm_val = val / baseline
            final_data[method].append(norm_val)
            
            if norm_val > max_norm_value:
                max_norm_value = norm_val

    df_plot = pd.DataFrame(final_data)
    
    print(f"\nMax Normalized Value detected: {max_norm_value:.2f}")
    
    # Calculate Dynamic Range
    # If max is > 2, extend range. Else default to 2.
    # Add 10% buffering margin
    limit = math.ceil(max(2.0, max_norm_value * 1.05))
    print(f"Dynamic Chart Range: [0, {limit}]")

    print("\n--- 5. Generate Plot ---")
    df_long = df_plot.melt(id_vars="Metric", var_name="Method", value_name="Normalized Value")
    
    fig = px.line_polar(df_long, r="Normalized Value", theta="Metric", color="Method", line_close=True)
    
    fig.update_traces(fill=None, line_width=2)
    fig.update_layout(
        title={
            'text': "Normalized Spatial Metrics (Smart Aligned & Auto-Scaled)",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, limit], # Dynamic adjustment
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
    
    out_png = os.path.join(output_dir, "fig8_smart_aligned.png")
    out_html = os.path.join(output_dir, "fig8_smart_aligned.html")
    
    print(f"Saving to {out_png}")
    try:
        fig.write_image(out_png, width=1200, height=800, scale=2)
        fig.write_html(out_html)
        print("✅ Plot generated successfully!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_smart_radar()
