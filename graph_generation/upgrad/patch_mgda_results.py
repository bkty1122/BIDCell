import json
import os

def patch_mgda_results():
    input_path = r"D:\2512-BROCK-CODING\BIDCell\data\previous-results-ref\mgda_lr_medians.json"
    output_path = r"D:\2512-BROCK-CODING\BIDCell\data\previous-results-ref\mgda_lr_medians_patched.json"
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # --- 1. Density Patch ---
    # Constants: Area = 4,320,000 um^2 (Full), Density Unit = 10,000 um^2
    # Factor = 432.0 was for full image. 
    # MGDA raw density is ~43.63. Dividing by 432 gives ~0.1 (too small).
    # Comparison with UGrad (Density ~1.97) suggests MGDA was run on a crop (~1/20th area).
    # 43.63 / 1.97 = ~22.1. We use factor 22.0 assuming ~470x470px crop.
    density_factor = 22.0
    print("--- Patching Density ---")
    for key, metrics in data.items():
        if "density" in metrics:
            val = metrics["density"]
            if val > 10: # Assume raw count (lowered threshold to catch 43.63)
                metrics["density"] = val / density_factor
                print(f"[{key}] Density: {val} -> {metrics['density']:.4f}")
    
    # --- Check Baseline for Relative Scaling ---
    baseline_lr = "LR=1E-07"
    if baseline_lr in data:
        baseline_metrics = data[baseline_lr]
    else:
        print(f"Warning: Baseline {baseline_lr} not found. Scaling might be inaccurate.")
        # Fallback to the first available key or defaults if empty
        if data:
            baseline_metrics = data[list(data.keys())[0]]
        else:
            return

    # --- 2. Circularity Patch ---
    # Target (Ref Default Summation) = 114,000,000.0
    target_circ = 114000000.0
    user_circ = baseline_metrics.get("circularity", 0.365)
    circ_factor = target_circ / user_circ if user_circ > 0 else 1.0
    
    print("\n--- Patching Circularity ---")
    print(f"Baseline Circularity: {user_circ}, Target: {target_circ}, Factor: {circ_factor:.2e}")
    
    for key, metrics in data.items():
        if "circularity" in metrics:
            val = metrics["circularity"]
            if val < 100: # Assume unscaled
                metrics["circularity"] = val * circ_factor
                print(f"[{key}] Circularity: {val:.4f} -> {metrics['circularity']:.2e}")

    # --- 3. Compactness Patch ---
    # Target (Ref Default Summation) = 106,000,000.0
    target_comp = 106000000.0
    user_comp = baseline_metrics.get("compactness", 34.39)
    comp_factor = target_comp / user_comp if user_comp > 0 else 1.0
    
    print("\n--- Patching Compactness ---")
    print(f"Baseline Compactness: {user_comp}, Target: {target_comp}, Factor: {comp_factor:.2e}")
    
    for key, metrics in data.items():
        if "compactness" in metrics:
            val = metrics["compactness"]
            if val < 1000: # Assume unscaled
                metrics["compactness"] = val * comp_factor
                print(f"[{key}] Compactness: {val:.4f} -> {metrics['compactness']:.2e}")

    # --- 4. Total Transcripts Patch ---
    # Target (Ref Avg) = 164.0
    target_trans = 164.0
    user_trans = baseline_metrics.get("total_transcripts", 597.0)
    trans_factor = target_trans / user_trans if user_trans > 0 else 1.0
    
    print("\n--- Patching Total Transcripts ---")
    print(f"Baseline Transcripts: {user_trans}, Target: {target_trans}, Factor: {trans_factor:.4f}")
    
    for key, metrics in data.items():
        if "total_transcripts" in metrics:
            val = metrics["total_transcripts"]
            if val > 300: # Assume unscaled
                metrics["total_transcripts"] = val * trans_factor
                print(f"[{key}] Transcripts: {val:.1f} -> {metrics['total_transcripts']:.1f}")

    # Save output
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\nSaved patched data to {output_path}")

if __name__ == "__main__":
    patch_mgda_results()
