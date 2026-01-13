
import json
import pandas as pd
import plotly.express as px
import numpy as np
import os
import glob
import re
import tifffile
import cv2

# Hardcoded reference max values for normalization
MAX_VALS = {
    'total_transcripts': 650,
    'total_genes': 70,
    'cell_area': 120, # standardized key
    'area': 120,
    'elongation': 0.7, 
    'compactness': 35,
    'sphericity': 0.8,
    'solidity': 0.9,
    'convexity': 1.0,
    'circularity': 0.6,
    'density': 10
}

LABELS_MAP = {
    "total_transcripts": "total_transcripts*",
    "total_genes": "total_genes*",
    "area": "cell_area*",
    "cell_area": "cell_area*",
    "elongation": "elongation",
    "compactness": "compactness*",
    "sphericity": "sphericity",
    "solidity": "solidity",
    "convexity": "convexity",
    "circularity": "circularity*",
    "density": "density*"
}

ORDERED_METRICS = [
    "density", "total_transcripts", "total_genes", "area", 
    "elongation", "compactness", "sphericity", 
    "solidity", "convexity", "circularity"
]

def compute_morphology_metrics(seg_path):
    try:
        seg = tifffile.imread(seg_path)
        cell_ids = np.unique(seg)
        cell_ids = cell_ids[cell_ids != 0]
        metrics = []

        if len(cell_ids) == 0:
            return metrics, 0

        for cid in cell_ids:
            mask = np.uint8(seg == cid)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            cnt = contours[0]
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            if area == 0: continue
            
            row = {"cell_id": int(cid), "area": area}
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            hull_perimeter = cv2.arcLength(hull, True)
            
            if hull_area > 0:
                solidity = area / hull_area
            else:
                solidity = 0
            row["solidity"] = solidity
            
            if perimeter > 0:
                convexity = hull_perimeter / perimeter
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                compactness = (perimeter ** 2) / area
            else:
                convexity = 0
                circularity = 0
                compactness = 0
            row["convexity"] = convexity
            row["circularity"] = circularity
            row["compactness"] = compactness
            row["sphericity"] = np.sqrt(circularity)

            if len(cnt) >= 5:
                try:
                    (x,y), (MA, ma), angle = cv2.fitEllipse(cnt)
                    major = max(MA, ma)
                    minor = min(MA, ma)
                    if major > 0:
                        elongation = minor / major
                    else:
                        elongation = 1.0
                except:
                    elongation = 1.0
            else:
                elongation = 1.0
            row["elongation"] = elongation
            
            metrics.append(row)
            
        return metrics, len(cell_ids)
    except Exception as e:
        print(f"    Error computing morphology: {e}")
        return [], 0

def compute_expression_metrics(cgm_path):
    if not os.path.exists(cgm_path):
        return []
    try:
        df = pd.read_csv(cgm_path, index_col=0)
        counts = df.sum(axis=1)
        n_genes = (df > 0).sum(axis=1)
        
        metrics = []
        for idx in df.index:
            metrics.append({
                "cell_id": int(idx) if str(idx).isdigit() else str(idx),
                "total_transcripts": int(counts[idx]),
                "total_genes": int(n_genes[idx])
            })
        return metrics
    except Exception as e:
        print(f"    Error computing expression: {e}")
        return []

def extract_metrics_from_folder(folder_path):
    # Try to find seg result
    # usually in folder/test_output/*_connected.tif OR folder/model_outputs/TIMESTAMP/test_output
    
    seg_files = glob.glob(os.path.join(folder_path, "test_output", "*_connected.tif"))
    if not seg_files:
        # Deep search
        seg_files = glob.glob(os.path.join(folder_path, "**", "*_connected.tif"), recursive=True)
    
    if not seg_files:
        return None
        
    seg_path = seg_files[0] # Take first
    morph_rows, count = compute_morphology_metrics(seg_path)
    if not morph_rows:
        return None
        
    morph_df = pd.DataFrame(morph_rows)
    metrics = {
        col: float(morph_df[col].median()) 
        for col in ["area", "elongation", "compactness", "sphericity", "solidity", "convexity", "circularity"]
        if col in morph_df.columns
    }
    
    metrics['density'] = float(count)
    
    # Expression
    cgm_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    cgm_files = [f for f in cgm_files if "expr" in f or "cell_gene" in f]
    
    if cgm_files:
        expr_rows = compute_expression_metrics(cgm_files[0])
        if expr_rows:
            expr_df = pd.DataFrame(expr_rows)
            metrics["total_transcripts"] = float(expr_df["total_transcripts"].median())
            metrics["total_genes"] = float(expr_df["total_genes"].median())
    
    return metrics

def process_method(root_dir, method_name):
    print(f"\nProcessing {method_name}...")
    results_dir = os.path.join(root_dir, f"{method_name}_results")
    out_json_path = os.path.join(results_dir, "lr_medians.json")
    os.makedirs(results_dir, exist_ok=True)
    
    sweep_dir = os.path.join(root_dir, "sweep_results")
    
    collected_data = {}
    
    # 1. Search for Sweep Results (Classified by LR)
    # Pattern: lr_{LR}_method_{method_name} e.g. lr_1e-05_method_ugrad
    if os.path.exists(sweep_dir):
        potential_dirs = glob.glob(os.path.join(sweep_dir, f"lr_*_method_{method_name}"))
        
        for pdir in potential_dirs:
            dirname = os.path.basename(pdir)
            # Extract LR
            parts = dirname.split('_')
            if len(parts) >= 2:
                # Assuming format lr_{LR}_method_{method}
                # lr_1e-05_method_ugrad -> parts: ['lr', '1e-05', 'method', 'ugrad']
                lr_str = parts[1] 
                
                # Standardize Label
                try:
                    lr_val = float(lr_str)
                    label = f"LR={lr_val:.0E}".replace("+", "")
                except:
                    label = f"LR={lr_str}"
                    
                print(f"  Found sweep dir: {dirname} -> {label}")
                metrics = extract_metrics_from_folder(pdir)
                if metrics:
                    collected_data[label] = metrics
                else:
                    print(f"    No valid metrics found in {pdir} (maybe empty?)")
                    
    # 2. If no sweep data found, try existing single result or existing lr_medians.json
    if not collected_data:
        # Check if we already have a populated lr_medians.json that might have been created by another script
        # But we want to overwrite/regenerate potentially.
        # Check for single-run morphology file
        print("  No sweep results found via folders. Checking single run data...")
        morph_path = os.path.join(results_dir, "morphology_metrics.json")
        if os.path.exists(morph_path):
            with open(morph_path, 'r') as f:
                d = json.load(f)
            df = pd.DataFrame(d)
            metrics = {
                col: float(df[col].median()) 
                for col in ["area", "elongation", "compactness", "sphericity", "solidity", "convexity", "circularity"]
                if col in df.columns
            }
            # Approximate other metrics if missing
            if 'cell_id' in df.columns:
                 # Check for CGM in 'example_data' if specific run cgm not found
                 pass # (Skipping deep search for single-run expression for brevity, relying on user putting it in valid locations)
            
            collected_data[method_name] = metrics

    # 3. Save Medians
    if collected_data:
        with open(out_json_path, 'w') as f:
            json.dump(collected_data, f, indent=4)
        print(f"  Saved medians to {out_json_path}")
        
        # 4. Generate Radar Plot
        generate_radar_plot(collected_data, method_name, root_dir)
    else:
        print(f"  No data found for {method_name}.")

def generate_radar_plot(data_dict, method_name, root_dir):
    # data_dict: { "LR=1E-05": {...}, "LR=1E-06": {...} }
    
    # Identify keys to plot
    all_keys = set()
    for m_dict in data_dict.values():
        all_keys.update(m_dict.keys())
        
    available_metrics = [m for m in ORDERED_METRICS if m in all_keys] 
    
    if not available_metrics:
        print("  No valid metrics to plot.")
        return

    plot_labels = [LABELS_MAP.get(m, m) for m in available_metrics]
    
    # Prepare Plotly Data
    plot_rows = []
    
    # Sort experiments
    exp_keys = list(data_dict.keys())
    try:
        exp_keys.sort(key=lambda x: float(x.split('=')[1]) if '=' in x else x, reverse=True) 
    except:
        exp_keys.sort()
        
    for exp_key in exp_keys:
        metrics = data_dict[exp_key]
        for m in available_metrics:
            val = metrics.get(m, 0)
            label = LABELS_MAP.get(m, m)
            
            # Normalize
            if "*" in label:
                max_v = MAX_VALS.get(m, 1.0)
                norm_val = val / max_v if max_v > 0 else 0
            else:
                norm_val = val
                
            plot_rows.append({
                "Metric": label,
                "Value": norm_val,
                "Experiment": exp_key
            })
            
    df = pd.DataFrame(plot_rows)
    
    key_name = "Learning Rate" if any("LR=" in k for k in exp_keys) else "Method"

    fig = px.line_polar(df, r="Value", theta="Metric", color="Experiment", line_close=True)
    
    fig.update_traces(fill=None)
    fig.update_layout(
        title={
            'text': f"{method_name} Learning Rate Comparison",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=12),
            ),
            angularaxis=dict(
                tickfont=dict(size=14),
            )
        ),
        legend=dict(
            title=dict(text=key_name),
            font=dict(size=14)
        ),
        font=dict(size=14)
    )

    out_dir = os.path.join(root_dir, "graph_generation", method_name)
    os.makedirs(out_dir, exist_ok=True)
    
    output_html = os.path.join(out_dir, f"{method_name}_lr_radar_plot.html")
    fig.write_html(output_html)
    
    output_png = os.path.join(out_dir, f"{method_name}_lr_radar_plot.png")
    try:
        fig.write_image(output_png, width=1200, height=800, scale=2)
        print(f"  Saved plot to {output_png}")
    except Exception as e:
        print(f"  Error saving PNG: {e}")

def main():
    root_dir = r"d:\2512-BROCK-CODING\BIDCell"
    methods = ["cagrad", "nashmtl", "ugrad", "sum"]
    
    for m in methods:
        process_method(root_dir, m)

if __name__ == "__main__":
    main()
