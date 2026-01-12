
import os
import sys
import glob
import re
import yaml
import json
import io
import shutil
import numpy as np
import pandas as pd
import tifffile
import cv2
import matplotlib.pyplot as plt
from bidcell.BIDCellModel import BIDCellModel
from bidcell.config import load_config
from bidcell.model.utils.utils import get_newest_id

# Helper to redirect stdout for parsing
class StdoutCapture:
    def __init__(self):
        self._stdout = sys.stdout
        self._string_io = io.StringIO()
    
    def __enter__(self):
        sys.stdout = self._string_io
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._stdout
        
    def get_output(self):
        return self._string_io.getvalue()

def parse_training_logs(log_text):
    """
    Parse the captured stdout for training metrics.
    Expected formats:
    Epoch[1/1], Step[0], Loss:2784.2937
    NE:0.5541, TC:195.2034, CC:1291.8763, OV:0.8927, PN:1295.7672
    Epoch = 1  lr = 1e-05
    """
    training_points = []
    learning_rates = []
    
    lines = log_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for learning rate
        # Format: Epoch = 1  lr = 1e-05
        lr_match = re.search(r"Epoch\s*=\s*(\d+)\s+lr\s*=\s*([0-9eE\.\-]+)", line)
        if lr_match:
            learning_rates.append({
                "epoch": int(lr_match.group(1)),
                "lr": float(lr_match.group(2))
            })
            
        # Check for Step loss
        # Format: Epoch[1/1], Step[0], Loss:2784.2937
        step_match = re.search(r"Epoch\[(\d+)/(\d+)\],\s*Step\[(\d+)\],\s*Loss:([0-9eE\.\-]+)", line)
        if step_match:
            current_point = {
                "epoch": int(step_match.group(1)),
                "max_epochs": int(step_match.group(2)),
                "step": int(step_match.group(3)),
                "total_loss": float(step_match.group(4))
            }
            
            # Look ahead for next line with breakdown
            # NE:0.5541, TC:195.2034, CC:1291.8763, OV:0.8927, PN:1295.7672
            if i + 1 < len(lines):
                breakdown_line = lines[i+1]
                parts = breakdown_line.split(',')
                if len(parts) >= 5 and "NE" in parts[0]:
                    try:
                        for part in parts:
                            key, val = part.strip().split(':')
                            current_point[key] = float(val)
                    except ValueError:
                        pass # Parsing error
            
            training_points.append(current_point)
            
        i += 1
        
    return training_points, learning_rates

def compute_morphology_metrics(seg_path):
    seg = tifffile.imread(seg_path)
    cell_ids = np.unique(seg)
    cell_ids = cell_ids[cell_ids != 0]
    
    metrics = []

    if len(cell_ids) == 0:
        return metrics

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
        
    return metrics

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

def plot_loss_curves(training_history, out_dir):
    if not training_history:
        return
        
    steps = [x['step'] for x in training_history]
    total_loss = [x['total_loss'] for x in training_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, total_loss, label='Total Loss', color='black', linewidth=2)
    
    keys = ['NE', 'TC', 'CC', 'OV', 'PN']
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    for key, color in zip(keys, colors):
        if key in training_history[0]:
            vals = [x[key] for x in training_history]
            plt.plot(steps, vals, label=key, color=color, alpha=0.7)
            
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves (Nash-MTL)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=300)
    plt.close()

def plot_lr_curve(learning_rates, out_dir):
    if not learning_rates:
        return
    
    epochs = [x['epoch'] for x in learning_rates]
    lrs = [x['lr'] for x in learning_rates]
    
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, lrs, 'o-', label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "lr_curve.png"), dpi=300)
    plt.close()

def main():
    nash_results_dir = os.path.join("D:\\2512-BROCK-CODING\\BIDCell", "nashmtl_results")
    if not os.path.exists(nash_results_dir):
        os.makedirs(nash_results_dir)
        
    print("Setting up example data...")
    if not os.path.exists("example_data"):
        BIDCellModel.get_example_data()
    
    # Isolate data for NashMTL
    nash_data_dir = "nash_data"
    if os.path.exists(nash_data_dir):
        shutil.rmtree(nash_data_dir)
    shutil.copytree("example_data", nash_data_dir)
    
    # Update config to point to isolated data
    params_file = "params_small_example.yaml"
    with open(params_file, 'r') as f:
        config = yaml.safe_load(f)
        
    config['files']['data_dir'] = os.path.join(nash_data_dir, "dataset_xenium_breast1_small")
    
    nash_config_file = "params_nash.yaml"
    with open(nash_config_file, 'w') as f:
        yaml.dump(config, f)

    print(f"Initializing model with {nash_config_file}...")
    model = BIDCellModel(nash_config_file)
    
    # Increase logging frequency to capture more data points
    print("Adjusting training frequency for maximum data capture (sample_freq=1)...")
    model.config.training_params.sample_freq = 1
    model.config.training_params.total_epochs = 5
    
    # Set Aggregation to NashMTL
    print("Using NashMTL Aggregation...")
    model.config.training_params.aggregation = "nashmtl"
    
    print("Running pipeline (Preprocessing, Training, Prediction)...")
    
    capture = StdoutCapture()
    with capture:
        try:
            model.run_pipeline()
        except SystemExit as e:
            print(f"Caught system exit: {e}")
        except Exception as e:
            print(f"Caught exception: {e}")
            import traceback
            traceback.print_exc()

    stdout_content = capture.get_output()
    print(stdout_content)

    if os.path.exists(nash_config_file):
        os.remove(nash_config_file)
    
    with open(os.path.join(nash_results_dir, "training_log.txt"), "w") as f:
        f.write(stdout_content)
        
    print("\nRetrieving metrics...")
    
    training_points, lrs = parse_training_logs(stdout_content)
    
    with open(os.path.join(nash_results_dir, "training_metrics.json"), "w") as f:
        json.dump({"history": training_points, "learning_rates": lrs}, f, indent=4)
        
    plot_loss_curves(training_points, nash_results_dir)
    plot_lr_curve(lrs, nash_results_dir)
    
    # Post-processing identification
    config = load_config(config_file) # Re-load to get clean paths
    if config.experiment_dirs.dir_id == "last":
        timestamp = get_newest_id(os.path.join(config.files.data_dir, "model_outputs"))
    else:
        timestamp = config.experiment_dirs.dir_id
        
    out_dir = os.path.join(config.files.data_dir, "model_outputs", timestamp)
    print(f"Using output from timestamp: {timestamp}")
    
    test_out_dir = os.path.join(out_dir, "test_output")
    connected_files = glob.glob(os.path.join(test_out_dir, "*_connected.tif"))
    
    if connected_files:
        seg_path = connected_files[0]
        print(f"Analyzing segmentation: {seg_path}")
        morph_metrics = compute_morphology_metrics(seg_path)
        with open(os.path.join(nash_results_dir, "morphology_metrics.json"), "w") as f:
            json.dump(morph_metrics, f, indent=4)
    else:
        print("Error: No segmentation output found.")

    # Expression Metrics
    # Check "cell_gene_matrices" (plural)
    cgm_dir = os.path.join(out_dir, "cell_gene_matrices")
    cgm_files = glob.glob(os.path.join(cgm_dir, "*.csv"))
    
    if cgm_files:
        cgm_path = cgm_files[0]
        print(f"Analyzing expression matrix: {cgm_path}")
        expr_metrics = compute_expression_metrics(cgm_path)
        with open(os.path.join(nash_results_dir, "expression_metrics.json"), "w") as f:
            json.dump(expr_metrics, f, indent=4)
    else:
        # Fallback to broad search if clean path fails
        all_csvs = glob.glob(os.path.join(out_dir, "**/*.csv"), recursive=True)
        cgm_candidates = [f for f in all_csvs if "cell_gene" in f or "cell_by_gene" in f or "matrix" in f]
        if cgm_candidates:
            print(f"Found candidate expression matrix (fallback): {cgm_candidates[0]}")
            expr_metrics = compute_expression_metrics(cgm_candidates[0])
            with open(os.path.join(nash_results_dir, "expression_metrics.json"), "w") as f:
                json.dump(expr_metrics, f, indent=4)
        else:
             print("Warning: No Cell-Gene Matrix CSV found.")
        
    print(f"Results saved to {nash_results_dir}")

if __name__ == "__main__":
    main()
