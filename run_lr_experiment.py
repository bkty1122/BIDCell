
import os
import sys
import glob
import re
import json
import time
import shutil
import numpy as np
import pandas as pd
import tifffile
import cv2
import matplotlib.pyplot as plt
from bidcell.BIDCellModel import BIDCellModel
from bidcell.config import load_config
from bidcell.model.utils.utils import get_newest_id

# -----------------------------------------------------------------------------
# Metric Computation Functions (Taken from example_ugrad_metrics.py)
# -----------------------------------------------------------------------------

def compute_morphology_metrics(seg_path):
    seg = tifffile.imread(seg_path)
    cell_ids = np.unique(seg)
    cell_ids = cell_ids[cell_ids != 0]
    
    metrics = []

    if len(cell_ids) == 0:
        return metrics, 0

    for cid in cell_ids:
        mask = np.uint8(seg == cid)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if area == 0: continue
        
        row = {"cell_id": int(cid), "area": area, "perimeter": perimeter}
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)
        
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0
        row["solidity"] = solidity
        
        if perimeter > 0:
            convexity = hull_perimeter / perimeter
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            compactness = (perimeter ** 2) / area
        else:
            convexity = 0
            circularity = 0
            compactness = 0
        row["convexity"] = convexity
        row["circularity"] = circularity
        row["compactness"] = compactness
        row["sphericity"] = np.sqrt(circularity)

        if len(cnt) >= 5:
            try:
                (x,y), (MA, ma), angle = cv2.fitEllipse(cnt)
                major = max(MA, ma)
                minor = min(MA, ma)
                if major > 0:
                    elongation = minor / major
                else:
                    elongation = 1.0
            except:
                elongation = 1.0
        else:
            elongation = 1.0
        row["elongation"] = elongation
        
        metrics.append(row)
        
    return metrics, len(cell_ids)

def compute_expression_metrics(cgm_path):
    if not os.path.exists(cgm_path):
        return []
        
    df = pd.read_csv(cgm_path, index_col=0)
    counts = df.sum(axis=1) # Sum over genes
    n_genes = (df > 0).sum(axis=1)
    
    metrics = []
    for idx in df.index:
        metrics.append({
            "cell_id": int(idx) if str(idx).isdigit() else str(idx),
            "total_transcripts": int(counts[idx]),
            "total_genes": int(n_genes[idx])
        })
    return metrics

# -----------------------------------------------------------------------------
# Main Experiment Loop
# -----------------------------------------------------------------------------

def main():
    # Define directories
    base_dir = r"D:\2512-BROCK-CODING\BIDCell"
    ugrad_results_dir = os.path.join(base_dir, "ugrad_results")
    if not os.path.exists(ugrad_results_dir):
        os.makedirs(ugrad_results_dir)
        
    config_file = "params_small_example.yaml"
    
    # Define Learning Rates to test
    # Matching the graph: 1E-05, 1E-06, 1E-07, 1E-08, 1E-09
    lrs = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    
    results = {}
    
    print("Starting Learning Rate Experiment...")
    print(f"Rates to test: {lrs}")
    
    for lr in lrs:
        lr_label = f"LR={lr:.1E}".replace("+", "") # Format as 1.0E-05
        # Clean up label if necessary to match exact graph format (10E-05 is unusual scientific notation, strictly 1E-05).
        # But user graph said "LR=10E-05" which is 10^-5 ?? Or is it 10 * 10^-5 = 10^-4?
        # Usually it means 1E-5. 
        # I will label it "LR=10E-XY" if I can match the format, but standard scientific is 1E-05.
        # User image shows "LR=10E-05". Let's assume they mean 1e-5.
        # Format "10E-05" is a bit weird (10 * 10^-5). Maybe they mean 1.0E-05.
        # Let's standardize on scientific notation 1E-05 for now.
        lr_label = f"LR={lr:.0E}".replace("+", "") 
        
        print(f"\n==========================================")
        print(f"Running Training with Learning Rate: {lr}")
        print(f"==========================================")
        
        # Initialize Model
        # We need to reload the model/config fresh every time
        model = BIDCellModel(config_file)
        
        # Override Parameters
        model.config.training_params.learning_rate = lr
        model.config.training_params.total_epochs = 5 # Use small epoch count for speed/demo, comparable to original script
        model.config.training_params.sample_freq = 1
        
        # Prepare Data if needed (should be cached, but calling get_example_data is safe)
        # BIDCellModel.get_example_data() 
        
        try:
            # Run Pipeline
            model.run_pipeline()
            
            # Identify Output Directory
            # Re-read config cleanly to get path settings
            config = load_config(config_file) 
            data_dir = config.files.data_dir
            timestamp = get_newest_id(os.path.join(data_dir, "model_outputs"))
            
            print(f"  Experiment completed. Output ID: {timestamp}")
            out_dir = os.path.join(data_dir, "model_outputs", timestamp)
            
            # -------------------------------------------------------
            # Metrics Extraction
            # -------------------------------------------------------
            current_metrics = {}
            
            # 1. Segmentation / Morphology
            test_out_dir = os.path.join(out_dir, "test_output")
            connected_files = glob.glob(os.path.join(test_out_dir, "*_connected.tif"))
            
            cell_count = 0
            if connected_files:
                seg_path = connected_files[0]
                morph_rows, cell_count = compute_morphology_metrics(seg_path)
                
                if morph_rows:
                    morph_df = pd.DataFrame(morph_rows)
                    # Compute Medians for the radar chart
                    for col in ["area", "elongation", "compactness", "sphericity", "solidity", "convexity", "circularity"]:
                         current_metrics[col] = float(morph_df[col].median())
                else:
                    print("  Warning: No cells found in segmentation.")
            else:
                 print("  Error: No segmentation output found.")
            
            # 2. Density (Proxy: Cell Count)
            current_metrics["density"] = float(cell_count)
            
            # 3. Expression
            # The structure observed is model_outputs/timestamp/test_output/... AND separate cell_gene_matrices/timestamp/expr_mat.csv
            # But BIDCellModel usually saves it to config.data_dir/cell_gene_matrices/timestamp
            
            # Let's try to look in the standard location relative to the data dir
            try:
                # Re-read config cleanly to be sure (already done above)
                cgm_base_dir = os.path.join(config.files.data_dir, "cell_gene_matrices", timestamp)
                cgm_files = glob.glob(os.path.join(cgm_base_dir, "*.csv"))
                
                if not cgm_files:
                    # Fallback: Search recursively in the output dir (for robustnees)
                     cgm_files = glob.glob(os.path.join(out_dir, "**/*.csv"), recursive=True)
                     cgm_files = [f for f in cgm_files if "cell_gene" in f or "cell_by_gene" in f or "matrix" in f]
            except:
                 cgm_files = []
            
            if cgm_files:
                expr_rows = compute_expression_metrics(cgm_files[0])
                if expr_rows:
                    expr_df = pd.DataFrame(expr_rows)
                    current_metrics["total_transcripts"] = float(expr_df["total_transcripts"].median())
                    current_metrics["total_genes"] = float(expr_df["total_genes"].median())
                else:
                    print("  Warning: Empty expression matrix.")
            else:
                print("  Warning: No expression matrix found.")
                
            # Store results
            results[lr_label] = current_metrics
            print(f"  Captured metrics: {current_metrics.keys()}")
            
        except Exception as e:
            print(f"  Error running experiment for LR {lr}: {e}")
            import traceback
            traceback.print_exc()
            
        # Optional sleep to ensure filesystem updates and timestamp distinctness
        time.sleep(2)

    # Save aggregated results
    out_json = os.path.join(ugrad_results_dir, "lr_medians.json")
    print(f"\nSaving aggregated results to {out_json}...")
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Done. You can now run the radar graph generation script.")

if __name__ == "__main__":
    main()
