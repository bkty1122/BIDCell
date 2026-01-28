
import os
import sys
import glob
import torch
import traceback
from bidcell.BIDCellModel import BIDCellModel
from bidcell.model.utils.utils import get_newest_id, sorted_alphanumeric

def find_latest_checkpoint(model_dir):
    """Finds the latest .pth checkpoint in the given directory."""
    try:
        files = glob.glob(os.path.join(model_dir, "*.pth"))
        if not files:
            return None, None, None
        
        # Sort files to find the latest
        files = sorted_alphanumeric(files)
        latest_file = files[-1]
        filename = os.path.basename(latest_file)
        
        # Expected format: epoch_X_step_Y.pth
        parts = filename.replace(".pth", "").split("_")
        # epoch is index 1, step is index 3
        epoch = int(parts[1])
        step = int(parts[3])
        
        return latest_file, epoch, step
    except Exception as e:
        print(f"Error parsing checkpoint filenames in {model_dir}: {e}")
        return None, None, None

def main():
    print("==========================================")
    print("BIDCell Prediction Only Script")
    print("==========================================\n")

    config_file = "params_mgda_small.yaml"
    
    # Initialize Model Wrapper
    try:
        model = BIDCellModel(config_file)
    except Exception as e:
        print(f"Error initializing model from {config_file}: {e}")
        return

    # Locate Data Directory
    data_dir = model.config.files.data_dir
    model_outputs_dir = os.path.join(data_dir, "model_outputs")
    
    if not os.path.exists(model_outputs_dir):
        print(f"Error: Model outputs directory not found at {model_outputs_dir}")
        return

    # Find Latest Experiment Timestamp
    try:
        timestamp = get_newest_id(model_outputs_dir)
        print(f"Found latest experiment timestamp: {timestamp}")
    except Exception as e:
        print(f"Error finding latest experiment: {e}")
        return

    experiment_path = os.path.join(model_outputs_dir, timestamp)
    models_dir = os.path.join(experiment_path, "models")
    
    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found at {models_dir}")
        return

    # 1. auto-detect checkpoint
    print(f"Scanning for checkpoints in: {models_dir}")
    ckpt_path, epoch, step = find_latest_checkpoint(models_dir)
    
    if ckpt_path is None:
        print("Error: No .pth checkpoint files found!")
        print("Training may have failed before saving any checkpoints.")
        return
        
    print(f"Found latest checkpoint: {os.path.basename(ckpt_path)}")
    print(f" > Epoch: {epoch}")
    print(f" > Step: {step}")
    
    # 2. Update configuration to use this checkpoint
    current_test_epoch = model.config.testing_params.test_epoch
    current_test_step = model.config.testing_params.test_step
    
    if current_test_epoch != epoch or current_test_step != step:
        print(f"Updating config: test_epoch {current_test_epoch} -> {epoch}, test_step {current_test_step} -> {step}")
        model.config.testing_params.test_epoch = epoch
        model.config.testing_params.test_step = step
        
        # Also ensure strict experiment ID is set so it doesn't try to look for 'last' again and fail if multiple exist
        model.config.experiment_dirs.dir_id = timestamp
    
    # 3. Run Prediction
    try:
        print("\nStarting Prediction Pipeline...")
        # We manually call predict() to ensure it runs
        model.predict()
        print("\n[SUCCESS] Prediction completed successfully.")
        
        # Verify output
        output_dir = os.path.join(experiment_path, "test_output")
        if os.path.exists(output_dir):
            print(f"Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
