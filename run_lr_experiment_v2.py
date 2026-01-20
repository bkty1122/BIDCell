
import os
import sys
import glob
import json
import time
import shutil
import numpy as np
import pandas as pd
import tifffile
import cv2
import traceback
import torch

# -----------------------------------------------------------------------------
# SAFETY CONFIGURATION
# -----------------------------------------------------------------------------
# 1. Set specific GPU (CUDA 3)
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

# 2. Memory Fragmentation Optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from bidcell.BIDCellModel import BIDCellModel
from bidcell.config import load_config
from bidcell.model.utils.utils import get_newest_id
from bidcell.download_utils import download_data

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def check_memory(required_gb=20.0):
    """Checks available CUDA memory on the visible device."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Using CPU.")
        return True

    # Check logical device 0 (which maps to physical CUDA_VISIBLE_DEVICES index)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        print(f" > Memory Check: {free_gb:.2f} GB FREE / {total_gb:.2f} GB TOTAL")
        
        if free_gb < required_gb:
            print(f"ERROR: Not enough memory! Required {required_gb} GB, found {free_gb:.2f} GB.")
            return False
        return True
    except Exception as e:
        print(f"Warning: Failed to check memory info ({e}). Proceeding...")
        return True

def load_marker_genes(csv_path):
    """Loads a set of unique genes from a marker CSV file."""
    if not os.path.exists(csv_path):
        print(f"Warning: Marker file not found at {csv_path}")
        return set()
    
    try:
        # Assuming format: CellType (index), Gene1, Gene2...
        df = pd.read_csv(csv_path, index_col=0)
        # Identify genes (columns) that have at least one marker entry (value > 0)
        # We sum down the columns; if sum > 0, it's a marker for at least one cell type.
        col_sums = df.sum(axis=0)
        genes = set(col_sums[col_sums > 0].index)
        print(f"  Loaded {len(genes)} marker genes from {os.path.basename(csv_path)}")
        return genes
    except Exception as e:
        print(f"  Error loading markers: {e}")
        return set()

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
    # Intersect matrix columns with marker sets
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

# -----------------------------------------------------------------------------
# Main Experiment Loop
# -----------------------------------------------------------------------------
def main():
    print("\n==========================================")
    print("BIDCell Learning Rate Experiment v2")
    print("Features: Memory Check, Full Metrics (Pos/Neg ExprsPct), Safe Save")
    print("==========================================\n")
    
    # 1. Setup Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ugrad_results_dir = os.path.join(base_dir, "full_data_results", "ugrad_results")
    if not os.path.exists(ugrad_results_dir):
        os.makedirs(ugrad_results_dir)
    
    results_json_path = os.path.join(ugrad_results_dir, "lr_medians.json")
    
    # 2. Check Initial Memory
    if not check_memory(required_gb=15.0):
        print("Initial memory check failed. Please ensure GPU 3 is free.")
        sys.exit(1)

    # 3. Load Config & Markers
    config_file = "params_paper.yaml"
    try:
        config = load_config(config_file)
        
        # Load markers
        # Paths in config are relative to the execution root usually.
        # We assume script is run from repo root D:\2512-BROCK-CODING\BIDCell
        pos_marker_path = os.path.abspath(config.files.fp_pos_markers)
        neg_marker_path = os.path.abspath(config.files.fp_neg_markers)
        
        pos_genes = load_marker_genes(pos_marker_path)
        neg_genes = load_marker_genes(neg_marker_path)
        
    except Exception as e:
        print(f"Error loading config or markers: {e}")
        traceback.print_exc()
        return

    # 4. Resume Capability
    results = {}
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, 'r') as f:
                results = json.load(f)
            print(f"Loaded existing results for: {list(results.keys())}")
        except:
            print("Could not load existing results. Starting fresh.")
            results = {}

    # 5. Define Learning Rates
    lrs = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    
    for lr in lrs:
        lr_label = f"LR={lr:.0E}".replace("+", "")
        
        if lr_label in results:
            print(f"Skipping {lr_label} - already completed.")
            continue
            
        print(f"\n>>> Starting Run for {lr_label}")
        
        # Memory Check before run
        if not check_memory(required_gb=12.0):
            print("  Memory low. Waiting 30s to drain...")
            time.sleep(30)
            if not check_memory(required_gb=10.0):
                print("  Critical memory shortage. Aborting experiment loop.")
                break
        
        try:
            # Re-init model
            model = BIDCellModel(config_file)
            model.config.training_params.learning_rate = lr
            model.config.training_params.total_epochs = 5
            model.config.training_params.sample_freq = 1
            
            # Identify output strategy
            data_dir = model.config.files.data_dir
            model_outputs_dir = os.path.join(data_dir, "model_outputs")
            if not os.path.exists(model_outputs_dir): os.makedirs(model_outputs_dir)
            existing_dirs = set(os.listdir(model_outputs_dir))
            
            # --- RUN PIPELINE ---
            print("  Running pipeline...")
            model.run_pipeline()
            # --------------------
            
            # Identify new output
            current_dirs = set(os.listdir(model_outputs_dir))
            new_dirs = current_dirs - existing_dirs
            if new_dirs:
                timestamp = list(new_dirs)[0]
                out_dir = os.path.join(model_outputs_dir, timestamp)
            else:
                timestamp = get_newest_id(model_outputs_dir)
                out_dir = os.path.join(model_outputs_dir, timestamp)
            print(f"  Output generated at: {out_dir}")

            # --- PROCESS METRICS ---
            current_metrics = {}
            
            # Morphology
            test_out_dir = os.path.join(out_dir, "test_output")
            connected_files = glob.glob(os.path.join(test_out_dir, "*_connected.tif"))
            
            density_val = 0.0
            if connected_files:
                seg_path = connected_files[0]
                pix_um = model.config.affine.target_pix_um or 1.0
                m_rows, _, den = compute_morphology_metrics(seg_path, pix_um)
                
                if m_rows:
                    df_morph = pd.DataFrame(m_rows)
                    for c in ["area", "elongation", "compactness", "sphericity", "solidity", "convexity", "circularity"]:
                        current_metrics[c] = float(df_morph[c].median())
                density_val = den
            current_metrics["density"] = float(density_val)
            
            # Expression (With Pos/Neg Pct)
            # Find CGM
            cgm_files = glob.glob(os.path.join(model.config.files.data_dir, "cell_gene_matrices", timestamp, "*.csv"))
            if not cgm_files:
                cgm_files = glob.glob(os.path.join(out_dir, "**/*.csv"), recursive=True)
                
            if cgm_files:
                e_rows = compute_expression_metrics(cgm_files[0], pos_genes=pos_genes, neg_genes=neg_genes)
                if e_rows:
                    df_expr = pd.DataFrame(e_rows)
                    current_metrics["total_transcripts"] = float(df_expr["total_transcripts"].median())
                    current_metrics["total_genes"] = float(df_expr["total_genes"].median())
                    current_metrics["positive exprsPct"] = float(df_expr["positive_exprsPct"].median())
                    current_metrics["negative exprsPct"] = float(df_expr["negative_exprsPct"].median())
            else:
                print("  Warning: No expression matrix found.")

            # --- SAVE RESULTS ---
            results[lr_label] = current_metrics
            with open(results_json_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"  [success] Metrics saved for {lr_label}")
            print(f"  Metrics captured: {list(current_metrics.keys())}")
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            time.sleep(2)
            
        except Exception as e:
            print(f"  [error] Experiment failed for {lr_label}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()
    
    print("\nExperiment Sequence Completed.")

if __name__ == "__main__":
    main()
