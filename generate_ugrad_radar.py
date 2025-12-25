
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Paths
    morph_path = r"d:\2512-BROCK-CODING\BIDCell\ugrad_results\morphology_metrics.json"
    cgm_dir = r"d:\2512-BROCK-CODING\BIDCell\example_data\dataset_xenium_breast1_small\cell_gene_matrices"
    
    # Find latest cgm
    timestamps = sorted([d for d in os.listdir(cgm_dir) if os.path.isdir(os.path.join(cgm_dir, d))])
    if not timestamps:
        print("No cell gene matrices found.")
        return
    
    latest_ts = timestamps[-1]
    cgm_path = os.path.join(cgm_dir, latest_ts, "expr_mat.csv")
    print(f"Using CGM from: {cgm_path}")
    
    # Load Data
    with open(morph_path, 'r') as f:
        morph_data = json.load(f)
    morph_df = pd.DataFrame(morph_data)
    
    if os.path.exists(cgm_path):
        expr_df = pd.read_csv(cgm_path, index_col=0)
        # Compute expression metrics
        expr_df['total_transcripts'] = expr_df.sum(axis=1)
        expr_df['total_genes'] = (expr_df > 0).sum(axis=1)
        
        # Merge
        # Ensure cell_id types match
        morph_df['cell_id'] = morph_df['cell_id'].astype(int)
        expr_df.index = expr_df.index.astype(int)
        
        df = morph_df.merge(expr_df[['total_transcripts', 'total_genes']], left_on='cell_id', right_index=True, how='inner')
    else:
        print("CGM file not found, using only morphology.")
        df = morph_df
        
    # Calculate Density
    if 'total_transcripts' in df.columns and 'area' in df.columns:
        df['density'] = df['total_transcripts'] / df['area']
        
    # Select Metrics for Radar
    # Axes from image: total_transcripts, total_genes, cell_area, elongation, compactness, sphericity, solidity, convexity, circularity, density
    
    metrics = {
        'total_transcripts': 'total_transcripts',
        'total_genes': 'total_genes',
        'cell_area': 'area',
        'elongation': 'elongation',
        'compactness': 'compactness',
        'sphericity': 'sphericity',
        'solidity': 'solidity',
        'convexity': 'convexity',
        'circularity': 'circularity',
        'density': 'density'
    }
    
    # Filter available metrics
    available_metrics = {k: v for k, v in metrics.items() if v in df.columns}
    
    # Compute Aggregates (Median)
    medians = df[list(available_metrics.values())].median()
    
    # Prepare Data for Plot
    # We need to normalize to 0-1 for the radar chart to look good.
    # Since we only have one dataset, we will Normalize by Max (or 99th percentile) of the raw data 
    # to show where the median sits relative to the population range? 
    # OR, we just normalize the Medians themselves to fit the chart?
    # Let's normalize by an assumed "Max" plausible value or just self-normalize to 1.0?
    # If we self-normalize to 1.0, we get a perfect circle of 1s. That's boring.
    # The user's chart has variation.
    # Let's Normalize by the Mean + 2 StdDev? Or Max?
    # Let's use the 95th percentile of the population as the "1.0" mark.
    
    values = []
    labels = []
    
    # Ordering roughly matching the image
    ordered_keys = [
        'total_transcripts', 'total_genes', 'cell_area', 'elongation', 
        'compactness', 'sphericity', 'solidity', 'convexity', 'circularity', 'density'
    ]
    
    print("\n--- Metrics Summary ---")
    for key in ordered_keys:
        if key in available_metrics:
            col = available_metrics[key]
            med = medians[col]
            p99 = df[col].quantile(0.99)
            norm_val = med / p99 if p99 > 0 else 0
            
            # Correction: Some metrics like Solidity are naturally 0-1 (max 1).
            # If Median Solidity is 0.9, and Max is 1.0, it plots at 0.9.
            # For Area, Median might be 100, Max 500. Plots at 0.2.
            # This seems like a reasonable way to shape the ugrad polygon.
            
            if key in ['solidity', 'convexity', 'circularity', 'sphericity', 'elongation']:
                norm_val = med # These are ratios 0-1 (mostly)
            else:
                norm_val = med / p99 if p99 > 0 else 0
            
            values.append(norm_val)
            labels.append(key)
            print(f"{key}: Median={med:.2f}, P99={p99:.2f}, Norm={norm_val:.2f}")

    # Close the loop
    values += values[:1]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], labels, color='black', size=10)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1.1)
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', label='Aligned-MTL (ugrad)')
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title("Normalized Spatial Metrics (ugrad)", size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    out_path = os.path.join(r"d:\2512-BROCK-CODING\BIDCell\ugrad_results", "ugrad_radar_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nRadar chart saved to: {out_path}")

if __name__ == "__main__":
    main()
