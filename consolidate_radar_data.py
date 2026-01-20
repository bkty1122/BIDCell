import pandas as pd
import json
import os

# Paths
base_dir = r"D:\2512-BROCK-CODING\BIDCell\data\previous-results-ref"
csv_path = os.path.join(base_dir, "radar-graph-comparasion-raw-data.csv")
json_path = os.path.join(base_dir, "ugrad.json")
output_path = os.path.join(base_dir, "consolidated_radar_data.json")

# Mapping from CSV metric names to simpler JSON keys (matching ugrad.json)
metric_map = {
    "Cell Area": "area",
    "Circularity": "circularity",
    "Compactness": "compactness",
    "Convexity": "convexity",
    "Density": "density",
    "Elongation": "elongation",
    "Solidity": "solidity",
    "Sphericity": "sphericity",
    "Total Genes": "total_genes",
    "Total Transcripts": "total_transcripts"
}

def consolidate():
    print(f"Reading CSV from: {csv_path}")
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Identify the value column (3rd column, index 2)
    # Columns: Metric Category, Method, 1.00E-07
    if len(df.columns) < 3:
        print(f"Error: CSV has unexpected number of columns: {df.columns}")
        return
    
    value_col = df.columns[2]
    print(f"Using value column: '{value_col}'")

    consolidated_data = {}

    # Process CSV Data
    # Group by Method
    methods = df['Method'].unique()
    for method in methods:
        method_df = df[df['Method'] == method]
        method_metrics = {}
        
        for _, row in method_df.iterrows():
            metric_cat = row['Metric Category']
            val = row[value_col]
            
            # Map metric name
            key = metric_map.get(metric_cat, metric_cat)
            method_metrics[key] = float(val)
        
        consolidated_data[method] = method_metrics

    print(f"Loaded {len(methods)} methods from CSV: {methods}")

    # Read JSON
    print(f"Reading JSON from: {json_path}")
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    # Process JSON Data (UGrad results)
    # Structure: "LR=..." -> { metrics }
    for lr_key, metrics in json_data.items():
        # Create a unique method name, e.g., "ugrad_LR=1E-05"
        # Check if the key already has 'ugrad' in it? It doesn't seem to.
        new_method_name = f"ugrad_{lr_key}"
        consolidated_data[new_method_name] = metrics

    print(f"Loaded {len(json_data)} entries from JSON.")

    # Write Output
    print(f"Writing consolidated data to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(consolidated_data, f, indent=4)
    
    print("Done. Data consolidated successfully.")

if __name__ == "__main__":
    consolidate()
