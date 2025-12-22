
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def plot_radar_chart(metrics_data, output_path):
    """
    Plots a radar chart (spider graph) from normalized metrics.
    Metrics should be in the range [0, 1] usually.
    Since raw metrics vary widely, we will Min-Max scale them if we had multiple datasets.
    For a single dataset, we just plot the raw values, but Radar Charts work best when axes have similar scales
    OR we have multiple labeled axes with different scales.
    
    Given the user input is a single set of morphology metrics from one run,
    we will plot the MEAN value of each metric.
    """
    
    # Calculate Mean of each metric
    # metrics_data is a list of dicts: [{'area': 100, ...}, ...]
    if not metrics_data:
        print("No data to plot.")
        return

    keys = list(metrics_data[0].keys())
    # Filter numeric keys
    keys = [k for k in keys if k not in ['cell_id']]
    
    means = {}
    for k in keys:
        values = [d[k] for d in metrics_data]
        means[k] = np.mean(values)
        
    print("Mean Metrics:", means)
    
    # Normalize for Radar Chart aesthetic if values vary wildly?
    # Actually, a standard Radar chart usually plots multiple groups (e.g. Models) on the SAME axes.
    # If we only have ONE group (the current run), a radar chart is just a shape.
    # The user asked for "similar graphs", implying comparison.
    # But we only have one result set in `ugrad_results`.
    # We will plot this one result set. To make it look like a "chart", we typically normalize values to 0-1 range
    # if we know the bounds. We don't.
    # So we will standardise them to relative 0-1 against an arbitrary max (e.g. 2x mean?) or just log scale?
    # Usually Radar charts for cell metrics use Min-Max scaling across methods.
    # Since we lack other methods, we will just plot the raw values on different axes (Parallel Coordinates in Circle)
    # OR we just plot them normalized by their own magnitude to fit the chart, with labels indicating value.
    
    # Better approach for single dataset: Normalizd against "Ideal" or just raw values?
    # Let's try to plot Raw Values but using different scales for each axis is hard in basic matplotlib radar.
    # Standard trick: Normalize everything to 0-1 and label axes with min/max.
    
    # For this task, we'll implement a simple Normalized Radar Chart 
    # where we assume some standard "max" observed in biology or just 1.0 for ratios (Solidity, Circularity).
    # For Area, we scaler by max observed?
    
    labels = list(means.keys())
    stats = list(means.values())
    
    # Separate ratios [0-1] from unbounded [Area, Perimeter]
    ratio_keys = ['solidity', 'convexity', 'circularity', 'compactness', 'sphericity', 'elongation'] 
    # Note: Compactness can be > 1 depending on definition (P^2/A). 
    # Elongation is usually >= 1.
    # Solidity, Convexity, Sphericity are <= 1.
    
    # Let's just normalize to max 1 for visualization relative to "1.0"
    # Or just plot what we have.
    
    # Number of variables
    num_vars = len(labels)
    
    # Cpmpute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # The chart is a circle, so we need to "close the loop"
    stats += stats[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], labels, color='grey', size=8)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
    plt.ylim(0, 1) # Most ratios are 0-1. Area will be off-chart.
    
    # Check if we have huge values (Area)
    # If we do, we should normalize 'stats' for the PLOT, but keep labels real?
    # For "Area", let's normalize by dividing by max value in dataset or 100?
    # Let's skip Area/Perimeter for the radar plot to focus on SHAPE descriptors (ratios) which fit the 0-1 chart better.
    
    shape_stats = []
    shape_labels = []
    for k, v in means.items():
        if k in ratio_keys:
            if k in ['solidity', 'convexity', 'sphericity']:
                shape_stats.append(v)
                shape_labels.append(k)
            elif k == 'elongation':
                # Inverse elongation (Minor/Major) is 0-1. Code computed Minor/Major.
                shape_stats.append(v)
                shape_labels.append(k)
            elif k == 'circularity': # Code computed 4piA/P^2 (<=1)
                shape_stats.append(v)
                shape_labels.append(k)
            elif k == 'compactness': # Code computed P^2/A (Big)
                 # Skip or normalize
                 pass
                 
    # Re-setup for shape stats only
    if not shape_stats:
        print("No ratio metrics found for radar chart.")
        return

    num_vars = len(shape_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    shape_stats += shape_stats[:1]
    angles += angles[:1]
    
    print(f"Plotting Radar with {num_vars} vars. Stats shape: {len(shape_stats)}, Angles shape: {len(angles)}")
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], shape_labels, color='black', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)
    
    ax.plot(angles, shape_stats, linewidth=1, linestyle='solid', label='Current Run')
    ax.fill(angles, shape_stats, 'b', alpha=0.1)
    
    plt.title('Morphology Metrics (Mean)', size=15, y=1.1)
    
    save_path = os.path.join(output_path, "radar_chart.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Radar chart saved to {save_path}")


def main():
    results_dir = "ugrad_results"
    params_file = os.path.join(results_dir, "morphology_metrics.json")
    
    if not os.path.exists(params_file):
        print(f"File not found: {params_file}")
        return
        
    with open(params_file, 'r') as f:
        metrics = json.load(f)
        
    plot_radar_chart(metrics, results_dir)

if __name__ == "__main__":
    main()
