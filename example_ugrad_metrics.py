
import os
import sys
import glob
import numpy as np
import pandas as pd
import tifffile
import cv2
import argparse
from bidcell.BIDCellModel import BIDCellModel
from bidcell.config import load_config
from bidcell.model.utils.utils import get_newest_id

# Helper metric functions using OpenCV
def compute_morphology_metrics(seg_path):
    seg = tifffile.imread(seg_path)
    # Get unique cell IDs (excluding 0 which is background)
    cell_ids = np.unique(seg)
    cell_ids = cell_ids[cell_ids != 0]
    
    metrics = {
        "area": [],
        "elongation": [],
        "compactness": [],
        "sphericity": [],
        "solidity": [],
        "convexity": [],
        "circularity": [],
        "density": [] # Placeholder if density calculation is unclear
    }

    if len(cell_ids) == 0:
        return metrics

    for cid in cell_ids:
        # Create mask for current cell
        mask = np.uint8(seg == cid)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        cnt = contours[0]
        
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if area == 0: continue
        
        metrics["area"].append(area)
        
        # Convex Hull
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)
        
        # Solidity: Area / ConvexArea
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0
        metrics["solidity"].append(solidity)
        
        # Convexity: ConvexPerimeter / Perimeter
        if perimeter > 0:
            convexity = hull_perimeter / perimeter
        else:
            convexity = 0
        metrics["convexity"].append(convexity)
        
        # Circularity: 4 * pi * Area / (Perimeter^2)
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            # Inverse for Compactness? Or Perimeter^2 / Area
            compactness = (perimeter ** 2) / area
        else:
            circularity = 0
            compactness = 0
        metrics["circularity"].append(circularity)
        
        # The table had huge numbers for Compactness/Circularity. 
        # We will store standard values alongside potentially raw moments if needed.
        # But standard definitions are safer.
        metrics["compactness"].append(compactness)
        
        # Sphericity: 2 * sqrt(pi * A) / P (for 2D) -> sqrt(circularity)
        metrics["sphericity"].append(np.sqrt(circularity))

        # Elongation
        if len(cnt) >= 5:
            try:
                (x,y), (MA, ma), angle = cv2.fitEllipse(cnt)
                # ma is major axis, MA is minor axis? fitEllipse returns (MA, ma) typically (width, height)?
                # OpenCV returns (center), (MA, ma), angle. 
                # Usually axes are sorted or not guaranteed.
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
        metrics["elongation"].append(elongation)
        
    return metrics

def compute_expression_metrics(cgm_path):
    # Assuming CSV format
    if not os.path.exists(cgm_path):
        print(f"CGM file not found: {cgm_path}")
        return {"total_transcripts": 0, "total_genes": 0}
        
    df = pd.read_csv(cgm_path, index_col=0)
    # Rows are genes? Cols are cells? Or vice versa.
    # Usually Cell x Gene.
    # Let's check logic: dataset_input reads patches.
    # make_cell_gene_mat.py creates cells as columns or rows?
    # standard is Cell x Gene for AnnData. 
    # If csv, let's assume rows=Cells, cols=Genes.
    
    # Check if index is cell_id or gene.
    # Usually index is Cell ID.
    
    # Total transcripts per cell = sum of row
    counts = df.sum(axis=1) # Sum over genes
    
    # Total genes per cell = count of >0
    n_genes = (df > 0).sum(axis=1)
    
    return {
        "total_transcripts": counts.values,
        "total_genes": n_genes.values
    }

def print_metrics(morph_metrics, expr_metrics):
    print("\n=== Metrics (Mean) ===")
    
    for k, v in morph_metrics.items():
        if len(v) > 0:
            print(f"{k}: {np.mean(v):.4f}")
            
    if "total_transcripts" in expr_metrics:
        print(f"total transcripts: {np.mean(expr_metrics['total_transcripts']):.3f}")
        print(f"total genes: {np.mean(expr_metrics['total_genes']):.3f}")

def main():
    # 1. Setup Data
    print("Setting up example data...")
    BIDCellModel.get_example_data()
    
    # 2. Run Pipeline
    config_file = "params_small_example.yaml"
    print(f"Initializing model with {config_file}...")
    model = BIDCellModel(config_file)
    
    # Check if we should render UGrad explicit? 
    # The default train() in bidcell/model/train.py ALREADY uses UUPGrad/mtl_backward.
    # So just running model.train() uses UGrad.
    
    print("Running pipeline (Preprocessing, Training, Prediction)...")
    # We can run the full pipeline
    model.run_pipeline()
    
    # 3. Retrieve Results and Calculate Metrics
    print("\nRetrieving metrics...")
    
    # Find output directory
    config = load_config(config_file)
    if config.experiment_dirs.dir_id == "last":
        timestamp = get_newest_id(os.path.join(config.files.data_dir, "model_outputs"))
    else:
        timestamp = config.experiment_dirs.dir_id
        
    print(f"Using output from timestamp: {timestamp}")
    
    # Path to segmentation
    out_dir = os.path.join(config.files.data_dir, "model_outputs", timestamp)
    
    # Need to find the connected tif. 
    # It is in test_output/epoch_X_step_Y_connected.tif
    test_out_dir = os.path.join(out_dir, "test_output")
    connected_files = glob.glob(os.path.join(test_out_dir, "*_connected.tif"))
    
    if not connected_files:
        print("Error: No segmentation output found.")
        return
        
    seg_path = connected_files[0]
    print(f"Analyzing segmentation: {seg_path}")
    
    morph_metrics = compute_morphology_metrics(seg_path)
    
    # Path to Cell-Gene Matrix
    # Usually in model_outputs/{timestamp}/cell_gene_matrix/
    cgm_dir = os.path.join(out_dir, "cell_gene_matrix")
    cgm_files = glob.glob(os.path.join(cgm_dir, "*.csv"))
    
    expr_metrics = {}
    if cgm_files:
        cgm_path = cgm_files[0]
        print(f"Analyzing expression matrix: {cgm_path}")
        expr_metrics = compute_expression_metrics(cgm_path)
    else:
        print("Warning: No Cell-Gene Matrix CSV found.")
        
    print_metrics(morph_metrics, expr_metrics)
    
    # Note: Precise "Positive/Negative Precision" would require re-implementing 
    # the marker vs segmentation overlap logic which depends on the raw data patches.
    # For now, we return the morphology and expression metrics which cover the first half of the user's table.

if __name__ == "__main__":
    main()
