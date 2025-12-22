
import os
import glob
import yaml
import copy
import shutil
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from bidcell.BIDCellModel import BIDCellModel
from bidcell.config import load_config
from bidcell.model.utils.utils import get_newest_id

# Settings
LEARNING_RATES = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
METHODS = ["ugrad", "sum"] # ugrad = Aligned-MTL, sum = Default summation
METHOD_LABELS = {"ugrad": "Aligned-MTL", "sum": "Default summation"}
BASE_CONFIG_PATH = "params_small_example.yaml"
OUTPUT_BASE = "sweep_results"

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f)

def run_experiment(lr, method):
    print(f"\n=== Running Experiment: LR={lr}, Method={method} ===")
    
    # Load base config
    config_data = load_yaml(BASE_CONFIG_PATH)
    
    # Modify parameters
    config_data['training_params']['learning_rate'] = lr
    config_data['training_params']['aggregation'] = method
    config_data['training_params']['total_epochs'] = 5
    
    # Set a unique data directory to avoid collisions and partial overwrites
    run_id = f"lr_{lr}_method_{method}"
    run_dir = os.path.join(OUTPUT_BASE, run_id)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        
    # We need to copy the example data to this new directory 
    # OR point the data directory to the original one but change the output dir.
    # BIDCell puts outputs in {data_dir}/model_outputs.
    # Changing data_dir implies copying the input data (images/transcripts).
    # Better strategy: Keep data_dir same, but control dir_id (output folder name).
    
    # Actually, config.files.data_dir is where it looks for inputs too.
    # So we should keep data_dir pointing to the data, but ensure unique dir_id.
    
    # config_data['files']['data_dir'] is likely "./example_data/dataset_xenium_breast1_small"
    # We will assume the input data is there.
    data_dir = config_data['files']['data_dir']
    
    # Set a specific experiment ID
    exp_id = f"exp_{lr}_{method}"
    config_data['experiment_dirs']['dir_id'] = exp_id
    
    # Save temp config
    temp_config_path = f"temp_config_{run_id}.yaml"
    save_yaml(config_data, temp_config_path)
    
    # Run Model
    try:
        model = BIDCellModel(temp_config_path)
        # We assume preprocess() has been run once on the data_dir.
        # If not, we should run it. example_small.py runs it.
        # To be safe, we can run preprocess() OR assume it's done. 
        # The first run will do it. Preprocess checks if files exist?
        # BIDCellModel.preprocess() often overwrites or re-runs.
        # If we re-run preprocess 10 times, it might be slow.
        # But 'segment_nuclei' etc might check existence.
        # Let's just run training and prediction if preprocess is already done.
        # How to check?
        # Check if 'expr_maps' exists in data_dir.
        
        if not os.path.exists(os.path.join(data_dir, "expr_maps")):
             print("Preprocessing...")
             model.preprocess()
        else:
             print("Skipping preprocessing (expr_maps found)...")
             
        print("Training...")
        model.train()
        print("Predicting...")
        model.predict()
        
        return os.path.join(data_dir, "model_outputs", exp_id)
        
    except Exception as e:
        print(f"Error executing {run_id}: {e}")
        return None
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def get_segmentation_image(output_dir):
    # Retrieve the final connected segmentation tif
    # location: {output_dir}/test_output/*_connected.tif
    if not output_dir or not os.path.exists(output_dir):
        return None
    
    test_out = os.path.join(output_dir, "test_output")
    files = glob.glob(os.path.join(test_out, "*_connected.tif"))
    if files:
        return files[0]
    return None

def main():
    # Ensure raw data is available
    if not os.path.exists("example_data"):
        print("Getting example data...")
        BIDCellModel.get_example_data()

    results = {}
    
    # Run Sweeps
    for lr in LEARNING_RATES:
        results[lr] = {}
        for method in METHODS:
            out_dir = run_experiment(lr, method)
            seg_path = get_segmentation_image(out_dir)
            results[lr][method] = seg_path
            
    # Generate Figure
    print("\nGenerating Figure...")
    
    n_rows = len(LEARNING_RATES)
    n_cols = len(METHODS) + 1 # +1 for Nuclei
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    # Get Nuclei Image (Raw)
    # config['files']['fp_nuclei'] usually "nuclei.tif" in data_dir
    config_data = load_yaml(BASE_CONFIG_PATH)
    data_dir = config_data['files']['data_dir']
    nuclei_path = os.path.join(data_dir, "nuclei.tif") # Resized/processed nuclei
    
    # If nuclei.tif doesn't exist, try original DAPI or run preprocess first
    if not os.path.exists(nuclei_path):
        # Maybe it's not processed yet?
        # Try finding the original:
        nuclei_path = config_data.get('files', {}).get('fp_dapi', None)

    nuclei_img = None
    if nuclei_path and os.path.exists(nuclei_path):
        nuclei_img = tifffile.imread(nuclei_path)
    
    for i, lr in enumerate(LEARNING_RATES):
        # Plot Methods
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            seg_path = results[lr].get(method)
            
            if seg_path and os.path.exists(seg_path):
                img = tifffile.imread(seg_path)
                # Colored segmentation
                ax.imshow(img, cmap='nipy_spectral', interpolation='nearest')
            else:
                ax.text(0.5, 0.5, "Not Found", ha='center')
            
            ax.axis('off')
            
            # Column Titles (Top Row only)
            if i == 0:
                ax.set_title(METHOD_LABELS[method], fontsize=12, fontweight='bold')
                
            # Row Labels (Left Column, but outside?)
            # We can use text label or ylabel
            if j == 0:
                ax.text(-0.1, 0.5, f"LR = {lr}", transform=ax.transAxes, 
                        va='center', ha='right', rotation=90, fontsize=12, fontweight='bold')

        # Plot Nuclei (Last Column)
        ax = axes[i, n_cols - 1]
        if nuclei_img is not None:
             # If nuclei image is HUGE, we might want to crop to same region as segmentation?
             # But segmentation usually covers the whole processed area.
             # Check sizes.
             if seg_path:
                 seg_sample = tifffile.imread(seg_path)
                 if seg_sample.shape != nuclei_img.shape:
                      # Try to crop or resize?
                      # usually they match if preprocessed key
                      pass
             ax.imshow(nuclei_img, cmap='gray')
        else:
             ax.text(0.5, 0.5, "Nuclei Img Missing", ha='center')
        
        ax.axis('off')
        if i == 0:
            ax.set_title("Nuclei", fontsize=12, fontweight='bold')

    plt.tight_layout()
    out_fig = "figure_4_reproduction.png"
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {out_fig}")

if __name__ == "__main__":
    main()
