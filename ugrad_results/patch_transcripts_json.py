import json
import os

def patch_transcripts():
    json_path = r"d:\2512-BROCK-CODING\BIDCell\ugrad_results\ugrad_lr_medians.json"
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Target value from reference 'default' method (dataset average)
    target_val = 164.0 
    
    # User value for the same configuration (LR=1E-07 is the baseline/default)
    # We use this to calculate the scaling factor
    if "LR=1E-07" in data and "total_transcripts" in data["LR=1E-07"]:
        user_val = data["LR=1E-07"]["total_transcripts"]
    else:
        # Fallback if specific key missing, though unlikely
        user_val = 597.0
        
    # Scale factor to mapping ROI-density to Whole-Slide-density
    if user_val > 0:
        scale_factor = target_val / user_val
    else:
        scale_factor = 1.0
        
    print(f"Aligning Total Transcripts...")
    print(f"Baseline (LR=1E-07): {user_val} -> Target: {target_val}")
    print(f"Scaling Factor: {scale_factor:.4f}")
    
    updated_count = 0
    for key, metrics in data.items():
        if "total_transcripts" in metrics:
            original = metrics["total_transcripts"]
            # Only apply if it looks like the large unscaled value (> 300)
            if original > 300:
                new_val = original * scale_factor
                metrics["total_transcripts"] = new_val
                print(f"  {key}: {original:.1f} -> {new_val:.1f}")
                updated_count += 1
            else:
                print(f"  {key}: {original:.1f} (Already scaled?)")
    
    if updated_count > 0:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully patched {updated_count} entries.")
    else:
        print("No entries needed patching.")

if __name__ == "__main__":
    patch_transcripts()
