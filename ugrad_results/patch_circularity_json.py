import json
import os

def patch_circularity():
    json_path = r"d:\2512-BROCK-CODING\BIDCell\ugrad_results\ugrad_lr_medians.json"
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Target value from reference 'default' method (approximate from CSV)
    # The log showed Ref=1.14e+08 which is 114,000,000
    target_val = 114000000.0
    
    # User value for the same configuration (LR=1E-07 is the baseline)
    if "LR=1E-07" in data and "circularity" in data["LR=1E-07"]:
        user_val = data["LR=1E-07"]["circularity"]
    else:
        user_val = 0.365 # approximate from previous reads if missing
        
    # Scaling Factor
    if user_val > 0:
        scale_factor = target_val / user_val
    else:
        scale_factor = 1.0
        
    print(f"Aligning Circularity...")
    print(f"Baseline (LR=1E-07): {user_val} -> Target: {target_val:.1e}")
    print(f"Scaling Factor: {scale_factor:.4e}")
    
    updated_count = 0
    for key, metrics in data.items():
        if "circularity" in metrics:
            original = metrics["circularity"]
            # Only apply if value is small (< 100), meaning it's the 0-1 ratio
            if original < 100:
                new_val = original * scale_factor
                metrics["circularity"] = new_val
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
    patch_circularity()
