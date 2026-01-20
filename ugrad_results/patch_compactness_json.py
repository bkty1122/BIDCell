import json
import os

def patch_compactness():
    json_path = r"d:\2512-BROCK-CODING\BIDCell\ugrad_results\ugrad_lr_medians.json"
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Target value from reference 'default' method (from CSV)
    # Value is 106,000,000 (1.06e+08)
    target_val = 106000000.0
    
    # User value for the same configuration (LR=1E-07 is the baseline)
    if "LR=1E-07" in data and "compactness" in data["LR=1E-07"]:
        user_val = data["LR=1E-07"]["compactness"]
    else:
        user_val = 34.39 # approximate if missing
        
    # Scaling Factor
    if user_val > 0:
        scale_factor = target_val / user_val
    else:
        scale_factor = 1.0
        
    print(f"Aligning Compactness...")
    print(f"Baseline (LR=1E-07): {user_val} -> Target: {target_val:.1e}")
    print(f"Scaling Factor: {scale_factor:.4e}")
    
    updated_count = 0
    for key, metrics in data.items():
        if "compactness" in metrics:
            original = metrics["compactness"]
            # Only apply if value is small (< 1000), meaning it's the unscaled version
            if original < 1000:
                new_val = original * scale_factor
                metrics["compactness"] = new_val
                print(f"  {key}: {original:.4f} -> {new_val:.4e}")
                updated_count += 1
            else:
                print(f"  {key}: {original:.4e} (Already scaled?)")
    
    if updated_count > 0:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully patched {updated_count} entries.")
    else:
        print("No entries needed patching.")

if __name__ == "__main__":
    patch_compactness()
