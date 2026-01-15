import os
import subprocess
import sys
import shutil
import zipfile
from pathlib import Path

def setup_small_data(target_root_dir="./example_data"):
    """
    Copies the small example dataset from the repository's 'data' folder
    to the target directory (default: ./example_data), enabling usage of
    params_small_example.yaml.
    """
    # Assumption: this script is in <RepoRoot>/bidcell/download_utils.py
    # and the data is in <RepoRoot>/data
    current_file = Path(__file__).resolve()
    repo_root = current_file.parent.parent
    source_data_dir = repo_root / "data"
    
    target_path = Path(target_root_dir)
    
    if not source_data_dir.exists():
        print(f"Error: Source data directory not found at {source_data_dir}")
        return

    if not target_path.exists():
        os.makedirs(target_path)
        
    print(f"Setting up small example data from {source_data_dir} to {target_path}...")
    
    # Copy dataset_xenium_breast1_small
    src_ds = source_data_dir / "dataset_xenium_breast1_small"
    dst_ds = target_path / "dataset_xenium_breast1_small"
    
    if src_ds.exists():
        if dst_ds.exists():
             print(f"  {dst_ds} already exists.")
        else:
             print(f"  Copying {src_ds} -> {dst_ds}")
             shutil.copytree(src_ds, dst_ds)
    else:
        print(f"  Warning: {src_ds} not found in source.")
        
    # Copy sc_references
    src_ref = source_data_dir / "sc_references"
    dst_ref = target_path / "sc_references"
    
    if src_ref.exists():
        if dst_ref.exists():
             print(f"  {dst_ref} already exists.")
        else:
             print(f"  Copying {src_ref} -> {dst_ref}")
             shutil.copytree(src_ref, dst_ref)
    else:
        print(f"  Warning: {src_ref} not found in source.")
        
    print("Small data setup complete.")

DATASET_URL = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip"

def download_data(target_dir):
    """
    Downloads and extracts the Xenium Breast Cancer dataset to target_dir.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    zip_name = "Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip"
    zip_path = os.path.join(target_dir, zip_name)
    
    # Check if files already exist (simple check)
    expected_file = os.path.join(target_dir, "morphology_mip.ome.tif")
    if os.path.exists(expected_file):
        print(f"Data appears to be present at {target_dir}. Skipping download.")
        return

    print("="*60)
    print("Downloading Xenium Breast Cancer Dataset...")
    print(f"URL: {DATASET_URL}")
    print(f"Target: {zip_path}")
    print("This file is large (~15GB). Please wait...")
    print("="*60)

    # 1. Download
    try:
        # Try curl (Linux/Mac)
        subprocess.check_call(["curl", "-L", "-o", zip_path, DATASET_URL])
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Try wget (Linux)
            subprocess.check_call(["wget", "-O", zip_path, DATASET_URL])
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: neither 'curl' nor 'wget' could be found or they failed.")
            print("Please install one of them or download the file manually.")
            return

    if not os.path.exists(zip_path):
        print("Download failed: Zip file not found.")
        return

    # 2. Extract
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List files to see structure
            # Likely has a root folder
            file_list = zip_ref.namelist()
            root_folder = file_list[0].split('/')[0]
            
            zip_ref.extractall(target_dir)
            
        print("Extraction complete.")
        
        # 3. Organize
        # If extracted into a subdir (e.g. "outs"), move files up or leave them and user config logic handles it?
        # The URL implies "outs.zip", often 10x zips contain the files directly or in a folder.
        # Based on typical 10x structure, it might create a folder "Xenium_FFPE_Human_Breast_Cancer_Rep1_outs" or just "outs" or files directly.
        
        # Let's check what was extracted
        extracted_root = os.path.join(target_dir, root_folder)
        if os.path.exists(extracted_root) and os.path.isdir(extracted_root):
            print(f"Moving files from {extracted_root} to {target_dir}...")
            for item in os.listdir(extracted_root):
                s = os.path.join(extracted_root, item)
                d = os.path.join(target_dir, item)
                if os.path.exists(d):
                    # Remove existing to overwrite
                    if os.path.isdir(d):
                        shutil.rmtree(d)
                    else:
                        os.remove(d)
                shutil.move(s, d)
            os.rmdir(extracted_root)
            
        # Cleanup
        os.remove(zip_path)
        print("Cleanup done. Dataset is ready.")
        
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip archive.")
    except Exception as e:
        print(f"An error occurred during extraction/organization: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Xenium Dataset")
    parser.add_argument("--target_dir", type=str, default="./data_large/dataset_xenium_breast1", help="Directory to save data")
    parser.add_argument("--small", action="store_true", help="Setup small example data instead of downloading full dataset")
    args = parser.parse_args()
    
    if args.small:
        setup_small_data()
    else:
        download_data(args.target_dir)
