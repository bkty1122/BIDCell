
import os
import sys
import glob
import json
import shutil
import numpy as np
import pandas as pd
import tifffile
import cv2
import traceback
from bidcell.config import load_config

# -----------------------------------------------------------------------------
# Helper Functions (Copied from run_mgda_experiment.py for consistency)
# -----------------------------------------------------------------------------

def compute_morphology_metrics(seg_path, pixel_size_um=1.0):
    seg = tifffile.imread(seg_path)
    cell_ids = np.unique(seg)
    cell_ids = cell_ids[cell_ids != 0]
    
    metrics = []
    h, w = seg.shape
    total_area_um2 = h * w * (pixel_size_um ** 2)
    density = len(cell_ids) / (total_area_um2 / 10000.0) if total_area_um2 > 0 else 0

    if len(cell_ids) == 0:
        return metrics, 0, 0.0

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
        
    return metrics, len(cell_ids), density

def compute_expression_metrics(cgm_path, pos_genes=None, neg_genes=None):
    if not os.path.exists(cgm_path):
        return []
        
    df = pd.read_csv(cgm_path, index_col=0)
    # df: rows=cells, cols=genes
    
    counts = df.sum(axis=1) # Total transcripts per cell
    n_genes = (df > 0).sum(axis=1) # Genes detected per cell
    
    # Pre-calculate positive and negative sums if markers provided
    matrix_genes = set(df.columns)
    
    pos_cols = []
    if pos_genes:
        pos_cols = list(matrix_genes.intersection(pos_genes))
    
    neg_cols = []
    if neg_genes:
        neg_cols = list(matrix_genes.intersection(neg_genes))
        
    if pos_cols:
        pos_counts = df[pos_cols].sum(axis=1)
    else:
        pos_counts = pd.Series(0, index=df.index)
        
    if neg_cols:
        neg_counts = df[neg_cols].sum(axis=1)
    else:
        neg_counts = pd.Series(0, index=df.index)

    metrics = []
    for idx in df.index:
        total = int(counts[idx])
        p_count = int(pos_counts[idx])
        n_count = int(neg_counts[idx])
        
        row = {
            "cell_id": int(idx) if str(idx).isdigit() else str(idx),
            "total_transcripts": total,
            "total_genes": int(n_genes[idx]),
            "positive_exprsPct": (p_count / total) if total > 0 else 0.0,
            "negative_exprsPct": (n_count / total) if total > 0 else 0.0
        }
        metrics.append(row)
    return metrics

def load_marker_genes(csv_path):
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, index_col=0)
        col_sums = df.sum(axis=0)
        genes = set(col_sums[col_sums > 0].index)
        return genes
    except:
        return set()

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------

def main():
    print("==========================================")
    print("Finalize MGDA Experiment Results")
    print("==========================================\n")

    # 1. Config & Paths
    config_file = "params_mgda_small.yaml"
    config = load_config(config_file)
    
    # Base dirs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mgda_data_root = os.path.join(base_dir, "mgda_data") # Root where they want to move files
    mgda_results_dir = os.path.join(base_dir, "mgda_results") # Where JSON lives
    
    data_dir = config.files.data_dir
    model_outputs_dir = os.path.join(data_dir, "model_outputs")
    cgm_dir_root = os.path.join(data_dir, "cell_gene_matrices")

    # 2. Find Latest Timestamp
    try:
        from bidcell.model.utils.utils import get_newest_id
        timestamp = get_newest_id(model_outputs_dir)
        print(f"Latest Experiment Timestamp: {timestamp}")
    except:
        print("Could not determine latest timestamp.")
        return

    # Source Paths
    experiment_dir = os.path.join(model_outputs_dir, timestamp)
    test_output_dir = os.path.join(experiment_dir, "test_output")
    
    # Destination Paths
    # User asked to move to "mgda_data folder". We'll make a neat subdirectory there.
    dest_name = f"results_{timestamp}_LR_1E-09"
    dest_dir = os.path.join(mgda_data_root, dest_name)
    
    print(f"Source: {test_output_dir}")
    print(f"Destination: {dest_dir}")

    # 3. Move/Copy Files
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Copy TIFs from test_output
    if os.path.exists(test_output_dir):
        print("Copying segmentation results...")
        for f in glob.glob(os.path.join(test_output_dir, "*")):
            if os.path.isfile(f):
                shutil.copy(f, dest_dir)
            elif os.path.isdir(f):
                # Optional: Copy the subfolder if it contains valuable patch data
                # shutil.copytree(f, os.path.join(dest_dir, os.path.basename(f)), dirs_exist_ok=True)
                pass
        print(" > Copied segmentation images.")
    else:
        print("Warning: test_output directory not found!")
    
    # Copy CSVs (CGM)
    # Check cgm dir first
    cgm_timestamp_dir = os.path.join(cgm_dir_root, timestamp)
    copied_cgm = None
    
    if os.path.exists(cgm_timestamp_dir):
        print("Copying Cell-Gene Matrices from cgm dir...")
        for f in glob.glob(os.path.join(cgm_timestamp_dir, "*.csv")):
            shutil.copy(f, dest_dir)
            copied_cgm = f # keep ref for metrics
        print(" > Copied CGM files.")
    else:
        # Check inside experiment dir
        print("Searching for CSVs in experiment dir...")
        csvs = glob.glob(os.path.join(experiment_dir, "**/*.csv"), recursive=True)
        for f in csvs:
            shutil.copy(f, dest_dir)
            copied_cgm = f
            print(f" > Copied {os.path.basename(f)}")

    print(f"\n[Files Moved] You can find your results in:\n{dest_dir}\n")

    # 4. Calculate Metrics & Update JSON
    print("Calculating Metrics to update lr_medians.json...")
    
    # Load Markers
    pos_marker_path = os.path.abspath(config.files.fp_pos_markers)
    neg_marker_path = os.path.abspath(config.files.fp_neg_markers)
    pos_genes = load_marker_genes(pos_marker_path)
    neg_genes = load_marker_genes(neg_marker_path)
    
    current_metrics = {}
    
    # Morphology from the COPIED file (to verify it works)
    # Look for *_connected.tif or any .tif
    seg_file = None
    # We copied files to dest_dir
    possible_segs = glob.glob(os.path.join(dest_dir, "*_connected.tif"))
    if not possible_segs:
        possible_segs = glob.glob(os.path.join(dest_dir, "*.tif"))
        # Filter out 'fill.tif' if others exist, usually we want the full seg
        # Bidcell usually outputs 'epoch_X_step_Y.tif' or similar.
    
    if possible_segs:
        seg_file = possible_segs[0]
        # Prefer one that doesn't say 'fill' if possible, or maybe 'connected' is best
        # The user has 'test_output'.
        # Let's just pick one.
        print(f"Using segmentation file for metrics: {os.path.basename(seg_file)}")
        
        pix_um = config.affine.target_pix_um or 1.0
        m_rows, _, den = compute_morphology_metrics(seg_file, pix_um)
        
        if m_rows:
            df_morph = pd.DataFrame(m_rows)
            for c in ["area", "elongation", "compactness", "sphericity", "solidity", "convexity", "circularity"]:
                current_metrics[c] = float(df_morph[c].median())
        current_metrics["density"] = float(den)
    else:
        print("Warning: No segmentation TIFF found for metrics.")

    # Expression
    cgm_file = None
    if copied_cgm:
        cgm_file = os.path.join(dest_dir, os.path.basename(copied_cgm))
    
    if cgm_file and os.path.exists(cgm_file):
        print(f"Using CGM file for metrics: {os.path.basename(cgm_file)}")
        e_rows = compute_expression_metrics(cgm_file, pos_genes=pos_genes, neg_genes=neg_genes)
        if e_rows:
            df_expr = pd.DataFrame(e_rows)
            current_metrics["total_transcripts"] = float(df_expr["total_transcripts"].median())
            current_metrics["total_genes"] = float(df_expr["total_genes"].median())
            current_metrics["positive exprsPct"] = float(df_expr["positive_exprsPct"].median())
            current_metrics["negative exprsPct"] = float(df_expr["negative_exprsPct"].median())
    else:
        print("Warning: No Cell-Gene Matrix found for metrics.")

    # Update JSON
    results_json_path = os.path.join(mgda_results_dir, "lr_medians.json")
    
    # We assume this was LR=1E-09 as per the user's previous error log
    target_lr = "LR=1E-09" 
    
    if current_metrics:
        results = {}
        if os.path.exists(results_json_path):
            try:
                with open(results_json_path, 'r') as f:
                    results = json.load(f)
            except:
                pass
        
        results[target_lr] = current_metrics
        
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"\n[SUCCESS] Updated {results_json_path} for {target_lr}")
        print("Metrics Captured:")
        print(json.dumps(current_metrics, indent=2))
    else:
        print("\n[FAILED] No metrics could be calculated.")

if __name__ == "__main__":
    main()
