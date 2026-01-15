
import json
import numpy as np

def calculate_expression_medians():
    input_path = r"D:\2512-BROCK-CODING\BIDCell\ugrad_results\expression_metrics.json"
    
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
            
        transcripts = [d['total_transcripts'] for d in data]
        genes = [d['total_genes'] for d in data]
        
        median_transcripts = np.median(transcripts)
        median_genes = np.median(genes)
        
        print(f"Median Total Transcripts: {median_transcripts}")
        print(f"Median Total Genes: {median_genes}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    calculate_expression_medians()
