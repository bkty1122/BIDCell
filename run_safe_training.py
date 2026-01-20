
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

# -----------------------------------------------------------------------------
# SAFETY CONFIGURATION
# -----------------------------------------------------------------------------
# 1. Set specific GPU (CUDA 3)
# This maps physical GPU 3 to logical device 'cuda:0' for this script.
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

# 2. Memory Fragmentation Optimization
# Helps avoid 'reserved but unallocated' OOM errors.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from bidcell.BIDCellModel import BIDCellModel
from bidcell.config import load_config
from bidcell.model.utils.utils import get_newest_id
from bidcell.download_utils import download_data

# -----------------------------------------------------------------------------
# Helper Functions (Metrics)
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

# -----------------------------------------------------------------------------
# Main Safegaurd Script
# -----------------------------------------------------------------------------
def main():
    print("Starting Safe Training Script...")
    print(f"Targeting Server GPU 3 (Mapped to cuda:0 via env var)")
    
    # Run initial memory check
    if not check_memory(required_gb=15.0):
        print("Aborting to prevent OOM crash.")
        sys.exit(1)
        
    # Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ugrad_results_dir = os.path.join(base_dir, "full_data_results", "ugrad_results")
    if not os.path.exists(ugrad_results_dir):
        os.makedirs(ugrad_results_dir)
        
    results_json_path = os.path.join(ugrad_results_dir, "lr_medians.json")
    
    # Load previous results (Resume capability)
    results = {}
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, 'r') as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {results_json_path}.")
        except:
            print("Failed to load existing json. Starting fresh.")
            results = {}
            
    # Config
    config_file = "params_paper.yaml"
    
    # Ensure data download logic is skipped for now or robust
    try:
        config_check = load_config(config_file)
        # Check data here if needed
    except Exception as e:
        print(f"Config Error: {e}")
        return

    # Learning Rates
    lrs = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    
    for lr in lrs:
        lr_label = f"LR={lr:.0E}".replace("+", "")
        
        if lr_label in results:
            print(f"Skipping {lr_label} - already completed.")
            continue
            
        print(f"\n==========================================")
        print(f"Processing: {lr_label}")
        print(f"==========================================")
        
        # Check memory again before each run
        if not check_memory(required_gb=12.0):
             print("Memory level critical. Waiting 30s...")
             time.sleep(30)
             if not check_memory(required_gb=10.0):
                 print("Still insufficient memory. Aborting run.")
                 break
        
        try:
            # Re-init model
            model = BIDCellModel(config_file)
            model.config.training_params.learning_rate = lr
            model.config.training_params.total_epochs = 5
            model.config.training_params.sample_freq = 1
            
            # Helper to detect new folder
            data_dir = model.config.files.data_dir
            model_outputs_dir = os.path.join(data_dir, "model_outputs")
            if not os.path.exists(model_outputs_dir): os.makedirs(model_outputs_dir)
            
            existing_dirs = set(os.listdir(model_outputs_dir))
            
            # --- RUN PIPELINE ---
            model.run_pipeline()
            # --------------------
            
            # Detect output
            current_dirs = set(os.listdir(model_outputs_dir))
            new_dirs = current_dirs - existing_dirs
            if new_dirs:
                timestamp = list(new_dirs)[0]
                out_dir = os.path.join(model_outputs_dir, timestamp)
            else:
                timestamp = get_newest_id(model_outputs_dir)
                out_dir = os.path.join(model_outputs_dir, timestamp)
            
            # Process metrics
            current_metrics = {}
            test_out_dir = os.path.join(out_dir, "test_output")
            connected_files = glob.glob(os.path.join(test_out_dir, "*_connected.tif"))
            
            if connected_files:
                seg_path = connected_files[0]
                pix_um = model.config.affine.target_pix_um or 1.0
                m_rows, _, den = compute_morphology_metrics(seg_path, pix_um)
                
                if m_rows:
                    df = pd.DataFrame(m_rows)
                    for c in ["area", "elongation", "compactness", "sphericity", "solidity", "convexity", "circularity"]:
                        current_metrics[c] = float(df[c].median())
                current_metrics["density"] = float(den)
            else:
                print("Warning: No segmentation file found")
                
            # Expression
            try:
                # Look in standard place
                cgm_dir = os.path.join(model.config.files.data_dir, "cell_gene_matrices", timestamp)
                cgm_files = glob.glob(os.path.join(cgm_dir, "*.csv"))
                if not cgm_files:
                    cgm_files = glob.glob(os.path.join(out_dir, "**/*.csv"), recursive=True)
                    
                if cgm_files:
                   e_rows = compute_expression_metrics(cgm_files[0])
                   if e_rows:
                       e_df = pd.DataFrame(e_rows)
                       current_metrics["total_transcripts"] = float(e_df["total_transcripts"].median())
                       current_metrics["total_genes"] = float(e_df["total_genes"].median())
            except Exception as e:
                print(f"Error extracting expression metrics: {e}")

            # SAVE INCREMENTALLY
            results[lr_label] = current_metrics
            with open(results_json_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"SUCCESS: Saved data for {lr_label}")

            # Clean memory
            del model
            torch.cuda.empty_cache()
            time.sleep(2)

        except Exception as e:
            print(f"FAILED Run {lr_label}: {e}")
            traceback.print_exc()
            # Try to save what we have? (Not necessary if we save incrementally on success)
            # Maybe clean up memory harder
            torch.cuda.empty_cache()
            
    print("\nExperiment Sequence Completed.")

if __name__ == "__main__":
    main()
