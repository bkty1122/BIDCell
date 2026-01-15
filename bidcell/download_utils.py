import os
import subprocess
import sys
import shutil
import zipfile

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
    args = parser.parse_args()
    
    download_data(args.target_dir)
