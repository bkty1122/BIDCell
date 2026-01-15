import json
import os

def update_density():
    json_path = r"d:\2512-BROCK-CODING\BIDCell\ugrad_results\ugrad_lr_medians.json"
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Constants from established analysis
    # Image size: 1800 x 2400 pixels
    # Pixel size: 1.0 um/pixel
    # Area = 4,320,000 um^2
    # Density Unit = 100 x 100 um = 10,000 um^2
    # Normalization Factor = 4,320,000 / 10,000 = 432.0
    
    area_norm_factor = 432.0
    
    updated_count = 0
    for key, metrics in data.items():
        if "density" in metrics:
            original_val = metrics["density"]
            
            # If value is large (likely raw count > 100), assume it needs conversion
            # If value is small (< 10), assume it's already density
            if original_val > 50:
                new_val = original_val / area_norm_factor
                metrics["density"] = new_val
                print(f"{key}: Converted density {original_val} -> {new_val:.4f}")
                updated_count += 1
            else:
                print(f"{key}: Density {original_val} seems already corrected. Skipping.")
    
    if updated_count > 0:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully updated {updated_count} entries.")
    else:
        print("No entries needed updating.")

if __name__ == "__main__":
    update_density()
