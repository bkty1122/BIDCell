
import json

def update_lr_medians():
    json_path = r"D:\2512-BROCK-CODING\BIDCell\ugrad_results\lr_medians.json"
    
    # Values calculated from expression_metrics.json
    median_transcripts = 586.0
    median_genes = 56.0
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        updated_count = 0
        for key in data:
            # Update only if missing to avoid overwriting if they existed (though we know they don't)
            if "total_transcripts" not in data[key]:
                data[key]["total_transcripts"] = median_transcripts
            if "total_genes" not in data[key]:
                data[key]["total_genes"] = median_genes
            updated_count += 1
            
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Successfully updated {updated_count} entries in {json_path}")
        
    except Exception as e:
        print(f"Error updating file: {e}")

if __name__ == "__main__":
    update_lr_medians()
