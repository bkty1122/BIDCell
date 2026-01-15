
import json
import os
import pandas as pd
import numpy as np

def update_lr_medians_with_real_data():
    # Paths
    json_path = r"D:\2512-BROCK-CODING\BIDCell\ugrad_results\lr_medians.json"
    
    # Based on the folders I found in cell_gene_matrices
    # Timestamps map to LRs in the order they were run (based on file modification times or order)
    # The script ran LRs: 1e-5, 1e-6, 1e-7, 1e-8, 1e-9 sequentialy.
    # The timestamps found were:
    # 2026_01_15_00_03_17 -> 1e-5
    # 2026_01_15_00_11_52 -> 1e-6
    # 2026_01_15_00_19_29 -> 1e-7
    # 2026_01_15_00_26_53 -> 1e-8
    # 2026_01_15_00_33_52 -> 1e-9
    
    # Map LRs to Timestamps
    lr_map = {
        "LR=1E-05": "2026_01_15_00_03_17",
        "LR=1E-06": "2026_01_15_00_11_52", 
        "LR=1E-07": "2026_01_15_00_19_29",
        "LR=1E-08": "2026_01_15_00_26_53",
        "LR=1E-09": "2026_01_15_00_33_52"
    }
    
    base_cgm_dir = r"D:\2512-BROCK-CODING\BIDCell\example_data\dataset_xenium_breast1_small\cell_gene_matrices"
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        for lr_key, timestamp in lr_map.items():
            if lr_key not in data:
                print(f"Warning: {lr_key} not found in JSON. Skipping.")
                continue
                
            csv_path = os.path.join(base_cgm_dir, timestamp, "expr_mat.csv")
            
            if os.path.exists(csv_path):
                print(f"Processing {lr_key} from {csv_path}...")
                df = pd.read_csv(csv_path, index_col=0)
                
                # Calculate metrics
                total_transcripts = df.sum(axis=1).median()
                total_genes = (df > 0).sum(axis=1).median()
                
                print(f"  > Computed Transcripts: {total_transcripts}, Genes: {total_genes}")
                
                # Update JSON data
                data[lr_key]["total_transcripts"] = float(total_transcripts)
                data[lr_key]["total_genes"] = float(total_genes)
            else:
                print(f"Error: CSV not found for {lr_key} at {csv_path}")
                
        # Save updated JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"\nSuccessfully updated {json_path} with real expression data.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    update_lr_medians_with_real_data()
