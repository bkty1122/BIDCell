import json
import pandas as pd
import plotly.express as px
import os

def generate_normalized_comparison_radar_lr_with_mgda():
    """
    Generate radar plots comparing multiple methods normalized by the 'default' method (renamed to SUM).
    Iterates through all learning rates found in ugrad.json.
    Includes MGDA data from mgda_lr_medians.json.
    The 'default' method will be 1.0 for all metrics.
    """
    # Paths to data files
    csv_path = r"D:\2512-BROCK-CODING\BIDCell\data\previous-results-ref\radar-graph-comparasion-raw-data.csv"
    ugrad_json_path = r"D:\2512-BROCK-CODING\BIDCell\data\previous-results-ref\ugrad.json"
    mgda_json_path = r"D:\2512-BROCK-CODING\BIDCell\data\previous-results-ref\mgda_lr_medians_patched.json"
    output_dir = r"D:\2512-BROCK-CODING\BIDCell\graph_generation\upgrad"
    
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    if not os.path.exists(ugrad_json_path):
        print(f"Error: UGrad JSON file not found at {ugrad_json_path}")
        return
    if not os.path.exists(mgda_json_path):
        print(f"Error: MGDA JSON file not found at {mgda_json_path}")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {csv_path}...")
    # Read CSV data
    df_csv = pd.read_csv(csv_path)
    
    print(f"Loading data from {ugrad_json_path}...")
    # Read UGrad JSON data
    with open(ugrad_json_path, 'r') as f:
        ugrad_data = json.load(f)

    print(f"Loading data from {mgda_json_path}...")
    # Read MGDA JSON data
    with open(mgda_json_path, 'r') as f:
        mgda_data = json.load(f)
    
    # Base Metric mapping from JSON keys to CSV metric categories
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
    
    # Legend renaming map
    # Key: Method name in CSV or JSON
    # Value: Desired Legend Name
    legend_map = {
        'default': 'SUM',
        'amtl-min': 'AMTL(min-mode)',
        'amtl-median': 'AMTL(median-mode)',
        'stch-mu-0.0005': 'STCH(mu=0.0005)',
        # 'ugrad' and 'mgda' will be handled dynamically
    }
    
    # Define the order of metrics (matching the reference)
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
    
    # Create labels with no * to indicate normalization
    plot_labels = [
        "total_transcripts",
        "total_genes",
        "cell_area",
        "elongation",
        "compactness",
        "sphericity",
        "solidity",
        "convexity",
        "circularity",
        "density"
    ]
    
    # --- Process Base Methods Data First ---
    # Create a dictionary to store base method data (excluding ugrad for now)
    base_methods_data = {}
    
    for method in df_csv['Method'].unique():
        # Only process methods involved in our legend map to avoid clutter if CSV has others
        # But we need 'default' for normalization
        if method not in legend_map and method != 'default':
            continue
            
        base_methods_data[method] = {}
        method_df = df_csv[df_csv['Method'] == method]
        for _, row in method_df.iterrows():
            metric = row['Metric Category']
            value = row['1.00E-07']
            base_methods_data[method][metric] = value

    if 'default' not in base_methods_data:
        print("Error: 'default' method not found in CSV data. Cannot normalize.")
        return

    default_values = base_methods_data['default']

    # --- Iterate through each Learning Rate in UGrad Data ---
    for ugrad_key, ugrad_selected_data in ugrad_data.items():
        print(f"Processing {ugrad_key}...")
        
        # Current run data container
        current_methods_data = base_methods_data.copy()
        
        # Add current UGrad data
        ugrad_display_name = "UPGrad"
        current_methods_data[ugrad_display_name] = {}
        
        for json_key, csv_metric in metric_mapping.items():
            if json_key in ugrad_selected_data:
                current_methods_data[ugrad_display_name][csv_metric] = ugrad_selected_data[json_key]

        # Add current MGDA data (if matches the LR)
        mgda_display_name = "MGDA"
        if ugrad_key in mgda_data:
            mgda_selected_data = mgda_data[ugrad_key]
            current_methods_data[mgda_display_name] = {}
            for json_key, csv_metric in metric_mapping.items():
                if json_key in mgda_selected_data:
                    current_methods_data[mgda_display_name][csv_metric] = mgda_selected_data[json_key]
        else:
             print(f"Warning: Learning rate {ugrad_key} not found in MGDA data.")
             # Optionally skip MGDA for this plot or handle gracefully
             # For now, if we add it to the list, we need empty dict or skip adding to list.
             # We will handle it by only adding to plot list if it exists in data

        # Prepare normalized data structure
        normalized_data = {"Metric": plot_labels}
        
        # List of methods to include in the plot (internal names)
        # We need to map 'default' -> 'SUM' etc for the final DataFrame
        
        methods_to_plot_internal = ['default', 'amtl-min', 'amtl-median', 'stch-mu-0.0005', ugrad_display_name]
        if mgda_display_name in current_methods_data:
            methods_to_plot_internal.append(mgda_display_name)
        
        for method_internal in methods_to_plot_internal:
            # Determine the display name
            if method_internal in legend_map:
                method_display = legend_map[method_internal]
            else:
                method_display = method_internal # e.g. UPGrad, MGDA
            
            normalized_data[method_display] = []
            
            if method_internal not in current_methods_data:
                # Should not happen based on logic above
                print(f"Warning: {method_internal} not found in data.")
                continue

            for metric in ordered_metrics:
                if metric in current_methods_data[method_internal] and metric in default_values:
                    # Normalize by default value
                    val = current_methods_data[method_internal][metric]
                    norm_val = val / default_values[metric]
                    normalized_data[method_display].append(norm_val)
                else:
                    normalized_data[method_display].append(0)
        
        # Create DataFrame
        df = pd.DataFrame(normalized_data)
        
        # Melt for plotly
        df_long = df.melt(id_vars="Metric", var_name="method", value_name="Value")
        
        # Generate the plot
        fig = px.line_polar(df_long, r="Value", theta="Metric", color="method", line_close=True)
        
        # Styling
        fig.update_traces(fill=None)
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 2],
                    tickfont=dict(size=24), 
                ),
                angularaxis=dict(
                    tickfont=dict(size=28), 
                )
            ),
            legend=dict(
                title=dict(text="method", font=dict(size=22)), 
                font=dict(size=22) 
            ),
            font=dict(size=20) 
        )
        
        # Construct filename
        # Clean the key for filename (remove LR=, etc)
        # ugrad_key is like "LR=1E-05"
        clean_key = ugrad_key.replace("=", "-")
        output_png = os.path.join(output_dir, f"normalized_comparison_radar_with_mgda_{clean_key}.png")
        
        print(f"Saving plot to {output_png}")
        try:
            fig.write_image(output_png, width=1200, height=800, scale=2)
        except Exception as e:
            print(f"Error saving PNG for {ugrad_key}: {e}")

    print("All plots generated.")

if __name__ == "__main__":
    generate_normalized_comparison_radar_lr_with_mgda()
