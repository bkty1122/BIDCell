
import os
import glob
import yaml
import json
import copy
import pandas as pd
import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt
from bidcell.BIDCellModel import BIDCellModel

# --- METRIC FUNCTIONS ---

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
        
        row = {"cell_id": int(cid), "area": area}
        
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
    counts = df.sum(axis=1)
    n_genes = (df > 0).sum(axis=1)
    
    metrics = []
    for idx in df.index:
        metrics.append({
            "cell_id": int(idx) if str(idx).isdigit() else idx,
            "total_transcripts": int(counts[idx]),
            "total_genes": int(n_genes[idx])
        })
    return metrics

def get_aggregated_metrics(morph_metrics, expr_metrics):
    if not morph_metrics:
        return {}
        
    morph_df = pd.DataFrame(morph_metrics)
    
    if expr_metrics:
        expr_df = pd.DataFrame(expr_metrics)
        # Standardize cell_id
        morph_df['cell_id'] = morph_df['cell_id'].astype(int)
        expr_df['cell_id'] = expr_df['cell_id'].astype(int)
        
        df = morph_df.merge(expr_df, on='cell_id', how='inner')
    else:
        df = morph_df
        
    if 'total_transcripts' in df.columns and 'area' in df.columns:
        df['density'] = df['total_transcripts'] / df['area']
        
    # Desired keys
    keys = [
        'total_transcripts', 'total_genes', 'area', 'elongation', 
        'compactness', 'sphericity', 'solidity', 'convexity', 'circularity', 'density'
    ]
    
    # Compute Medians
    aggs = {}
    for k in keys:
        if k in df.columns:
            aggs[k] = float(df[k].median())
        else:
            aggs[k] = 0.0
            
    return aggs

# --- MAIN EXPERIMENT ---

from bidcell.download_utils import download_data, setup_small_data

def main():
    base_dir = r"D:\2512-BROCK-CODING\BIDCell"
    base_config_path = "params_small_example.yaml"

    # Pre-check for small data requirement to avoid load_config crash
    if "small" in base_config_path and not os.path.exists("./example_data/dataset_xenium_breast1_small"):
        print("Small example data missing. Setting it up...")
        setup_small_data()

    out_dir = os.path.join(base_dir, "full_data_results_small_loss_align", "ugrad_ablation_results")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Check for Data Existence (Full Paper Datasource)
    dapi_path = base_config['files']['fp_dapi']
    if not os.path.exists(dapi_path):
        target_dir = os.path.dirname(dapi_path)
        print(f"Data not found at {dapi_path}. Attempting auto-download...")
        download_data(target_dir)
        
        if not os.path.exists(dapi_path):
             print(f"Critical: Failed to obtain data at {dapi_path}. Exiting.")
             return
        
    # Ablation Settings
    ablations = {
        'ne': {'ne_weight': 0.0},
        'os': {'os_weight': 0.0},
        'cc': {'cc_weight': 0.0},
        'mu': {'ov_weight': 0.0},
        'pn': {'pos_weight': 0.0, 'neg_weight': 0.0}
    }
    
    # Store aggregated medians for each run
    experiment_results = {}
    
    for name, params in ablations.items():
        print(f"\n=== Running Ablation: {name} ===")
        
        # Modify Config
        config = copy.deepcopy(base_config)
        config['training_params']['aggregation'] = 'ugrad'
        config['training_params']['total_epochs'] = 2
        config['training_params']['model_freq'] = 60 # Save regularly to catch step 60
        config['training_params']['sample_freq'] = 1000
        
        # Sync Testing Params
        config['testing_params']['test_epoch'] = 2
        config['testing_params']['test_step'] = 60
        
        # Apply weights
        for k, v in params.items():
            config['training_params'][k] = v
            
        # Output Dir
        run_id = f"ablation_{name}"
        config['experiment_dirs']['dir_id'] = run_id
        
        print(f"Configured {name}: Epochs={config['training_params']['total_epochs']}, Freq={config['training_params']['model_freq']}, TestEpoch={config['testing_params']['test_epoch']}, TestStep={config['testing_params']['test_step']}")
        
        # Temp Config File
        temp_cfg = f"temp_ablation_{name}.yaml"
        with open(temp_cfg, 'w') as f:
            yaml.dump(config, f)
            
        try:
            # Run Model
            model = BIDCellModel(temp_cfg)
            # Assume preprocessing done by previous runs or default
            # Check for expr_maps
            if not os.path.exists(os.path.join(config['files']['data_dir'], "expr_maps")):
                model.preprocess()
            
            # Detect new output folder by checking before/after
            model_outputs_dir = os.path.join(config['files']['data_dir'], "model_outputs")
            if not os.path.exists(model_outputs_dir):
                os.makedirs(model_outputs_dir)
            existing_dirs = set(os.listdir(model_outputs_dir))
            
            model.train()
            
            current_dirs = set(os.listdir(model_outputs_dir))
            new_dirs = current_dirs - existing_dirs
            
            if new_dirs:
                # Assuming the most specific/relevant new dir is the one we want. 
                # If multiple created (unlikely), pick the one that looks like a timestamp or just the first.
                latest_id = list(new_dirs)[0]
                print(f" Detected new training output directory: {latest_id}")
            else:
                # Fallback if no new directory appears (shouldn't happen if training ran)
                from bidcell.model.utils.utils import get_newest_id
                latest_id = get_newest_id(model_outputs_dir)
                print(f" Warning: No new directory detected. Falling back to newest_id: {latest_id}")

            # Update config to point to the actual output directory
            config['experiment_dirs']['dir_id'] = latest_id
            
            # Write updated config back to temp file and reload the model
            with open(temp_cfg, 'w') as f:
                yaml.dump(config, f)
            
            # Reload model with updated config
            model = BIDCellModel(temp_cfg)
            
            model.predict()
            
            # Paths
            data_dir = config['files']['data_dir']
            
            # 1. Segmentation
            # In data_dir/model_outputs/run_id/test_output
            # Use `latest_id` instead of `ablation_name` run_id
            test_output = os.path.join(data_dir, "model_outputs", latest_id, "test_output")
            connected_tifs = glob.glob(os.path.join(test_output, "*_connected.tif"))
            
            morph_metrics = []
            if connected_tifs:
                print(f" processing segmentation: {connected_tifs[0]}")
                morph_metrics = compute_morphology_metrics(connected_tifs[0])
            else:
                print(" Warning: No segmentation file found.")
                
            # 2. Expression
            # In data_dir/cell_gene_matrices/run_id/expr_mat.csv
            cgm_path = os.path.join(data_dir, "cell_gene_matrices", latest_id, "expr_mat.csv")
            expr_metrics = []
            if os.path.exists(cgm_path):
                print(f" processing expression: {cgm_path}")
                expr_metrics = compute_expression_metrics(cgm_path)
            else:
                print(" Warning: No expression matrix found.")
                
            # Aggregate
            aggs = get_aggregated_metrics(morph_metrics, expr_metrics)
            experiment_results[name] = aggs
            print(f" Stats: {aggs}")
            
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(temp_cfg):
                os.remove(temp_cfg)
                
    # Save Results
    res_path = os.path.join(out_dir, "ablation_medians.json")
    with open(res_path, 'w') as f:
        json.dump(experiment_results, f, indent=4)
    print(f"\nExperiment Results saved to {res_path}")
    
    # --- PLOTTING ---
    plot_radar(experiment_results, out_dir)

def plot_radar(data, out_dir):
    if not data:
        return
        
    runs = list(data.keys())
    metrics = list(data[runs[0]].keys())
    
    # Normalize: "Median Scaled"
    # To match curve shape 0.8-1.0, we normalize by MAX value across runs for each metric.
    max_values = {}
    for m in metrics:
        vals = [data[r].get(m, 0) for r in runs]
        max_values[m] = max(vals) if vals and max(vals) > 0 else 1.0
        
    # Ordered keys for plot
    # Try to match the user's order if possible, or just standard
    # Image: total_transcripts, total_genes, cell_area, elongation, compactness, sphericity, solidity, convexity, circularity, density
    plot_keys = [
        'total_transcripts', 'total_genes', 'area', 'elongation', 
        'compactness', 'sphericity', 'solidity', 'convexity', 'circularity', 'density'
    ]
    # Filter valid keys
    plot_keys = [k for k in plot_keys if k in metrics]
    
    # Prepare Plot
    labels = [k.replace("area", "cell_area") + "*" for k in plot_keys] # Add * for visual match
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = {'ne': 'blue', 'os': 'red', 'cc': 'green', 'mu': 'purple', 'pn': 'orange'}
    
    for name in runs:
        values = []
        for k in plot_keys:
            raw = data[name].get(k, 0)
            norm = raw / max_values[k]
            values.append(norm)
        
        values += values[:1]
        
        c = colors.get(name, 'black')
        ax.plot(angles, values, label=name, color=c, linewidth=1.5)
        # ax.fill(angles, values, color=c, alpha=0.05) # Too cluttered with multiple fills
        
    plt.xticks(angles[:-1], labels, size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.9, 0.95, 1.0], ["0.9", "0.95", "1"], color="grey", size=7)
    plt.ylim(0.8, 1.02) # Zoom in to show differences!
    
    plt.legend(bbox_to_anchor=(1.3, 1.1), title="Ablation")
    plt.title("Normalized Spatial Metrics for Loss Ablation\nwith Aligned-MTL (Median-Scaled)", y=1.1)
    
    out_png = os.path.join(out_dir, "ablation_radar.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Radar Plot saved to {out_png}")

if __name__ == "__main__":
    main()
