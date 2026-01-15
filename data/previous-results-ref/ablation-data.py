import pandas as pd
import io
import re

text = """Metric Ablated Loss — Default Summation
ne os cc mu pn
total transcripts 284.473 269.354 268.484 194.777 272.840
total genes 73.039 72.396 71.620 63.441 73.352
cell area 202.186 196.835 187.260 161.461 192.340
elongation 0.811 0.831 0.760 0.792 0.785
compactness 9.872E+07 8.998E+07 9.221E+07 9.891E+07 8.844E+07
sphericity 0.601 0.598 0.589 0.597 0.595
solidity 0.809 0.819 0.807 0.818 0.830
convexity 0.923 0.930 0.927 0.930 0.930
circularity 1.092E+08 1.001E+08 1.049E+08 1.092E+08 9.947E+07
density 2.598 2.357 2.554 2.036 2.499
positive Precision 0.462 0.452 0.457 0.472 0.459
negative Precision 0.039 0.044 0.039 0.037 0.041
positive Recall 0.376 0.372 0.373 0.374 0.377
negative Recall 0.028 0.033 0.030 0.026 0.030
positive F1 0.409 0.404 0.405 0.412 0.410
negative F1 0.032 0.037 0.034 0.030 0.034
positive exprsPct 0.724 0.722 0.717 0.688 0.733
negative exprsPct 0.154 0.160 0.154 0.130 0.161
F1 purity 57.54% 56.95% 57.11% 57.84% 57.56% ; Metric Ablated Loss — Aligned-MTL (Median-Scaled)
ne os cc mu pn
total transcripts 297.233 306.840 273.648 214.523 213.833
total genes 76.351 77.228 72.826 65.981 65.092
cell area 202.798 204.886 195.723 167.186 161.568
elongation 0.799 0.793 0.783 0.789 0.793
compactness 8.717E+07 8.567E+07 8.322E+07 9.944E+07 1.026E+08
sphericity 0.602 0.596 0.590 0.603 0.593
solidity 0.810 0.817 0.790 0.803 0.815
convexity 0.919 0.926 0.920 0.930 0.931
circularity 1.004E+08 9.933E+07 9.543E+07 1.093E+08 1.136E+08
density 2.354 2.637 2.229 2.122 1.998
positive Precision 0.449 0.453 0.452 0.461 0.465
negative Precision 0.046 0.043 0.042 0.040 0.043
positive Recall 0.375 0.375 0.370 0.368 0.372
negative Recall 0.034 0.030 0.033 0.030 0.032
positive F1 0.405 0.406 0.402 0.404 0.409
negative F1 0.039 0.036 0.037 0.035 0.037
positive exprsPct 0.739 0.742 0.720 0.694 0.739
negative exprsPct 0.170 0.177 0.158 0.160 0.166
F1 purity 57.00% 57.11% 56.79% 56.94% 57.46% ; Metric Ablated Loss — Aligned-MTL (Min-Scaled)
ne os cc mu pn
total transcripts 283.996 212.705 145.829 277.388 315.766
total genes 74.786 65.702 54.430 74.004 77.657
cell area 190.141 167.767 127.438 198.073 204.777
elongation 0.822 0.805 0.808 0.848 0.791
compactness 9.696E+07 9.247E+07 1.196E+08 9.095E+07 9.474E+07
sphericity 0.595 0.597 0.598 0.592 0.598
solidity 0.818 0.817 0.831 0.823 0.825
convexity 0.925 0.929 0.941 0.928 0.926
circularity 1.099E+08 1.038E+08 1.246E+08 1.017E+08 1.054E+08
density 2.379 2.036 1.727 2.331 3.137
positive Precision 0.455 0.465 0.492 0.450 0.441
negative Precision 0.040 0.041 0.034 0.041 0.043
positive Recall 0.378 0.369 0.365 0.379 0.374
negative Recall 0.029 0.032 0.030 0.030 0.033
positive F1 0.409 0.406 0.413 0.403 0.400
negative F1 0.034 0.035 0.027 0.034 0.037
positive exprsPct 0.734 0.697 0.649 0.728 0.737
negative exprsPct 0.160 0.138 0.097 0.160 0.175
F1 purity 57.49% 57.18% 58.00% 57.38% 56.70% ; Metric Ablated Loss — STCH (μ = 0.0005)
ne os cc mu pn
total transcripts 329.349 158.859 260.207 177.008 241.510
total genes 77.720 56.910 72.701 60.335 70.004
cell area 208.157 136.812 184.674 152.031 166.521
elongation 0.828 0.812 0.776 0.776 0.795
compactness 8.970E+07 1.019E+08 1.061E+08 9.348E+07 1.032E+08
sphericity 0.599 0.598 0.593 0.593 0.599
solidity 0.807 0.828 0.798 0.819 0.808
convexity 0.923 0.937 0.921 0.932 0.928
circularity 1.023E+08 1.113E+08 1.135E+08 9.987E+07 1.140E+08
density 3.602 1.817 2.268 2.903 2.229
positive Precision 0.447 0.476 0.457 0.473 0.458
negative Precision 0.044 0.037 0.045 0.037 0.040
positive Recall 0.375 0.365 0.376 0.369 0.369
negative Recall 0.033 0.028 0.034 0.030 0.030
positive F1 0.404 0.408 0.408 0.420 0.404
negative F1 0.037 0.030 0.037 0.031 0.034
positive exprsPct 0.734 0.656 0.718 0.677 0.718
negative exprsPct 0.179 0.144 0.164 0.153 0.146
F1 purity 56.88% 57.44% 57.31% 57.51% 56.94%"""

sections = text.split(';')
data_list = []

for section in sections:
    lines = [line.strip() for line in section.strip().split('\n') if line.strip()]
    if not lines:
        continue
    
    # First line is method name
    method_name = lines[0].replace('Metric Ablated Loss — ', '')
    # Second line is headers
    headers = lines[1].split() # ne os cc mu pn
    
    # Remaining lines are metrics
    for line in lines[2:]:
        parts = line.split()
        # The last 5 parts are the values
        values = parts[-5:]
        # The preceding parts are the metric name
        metric_name = " ".join(parts[:-5])
        
        row = {
            'Method': method_name,
            'Metric': metric_name,
        }
        for h, v in zip(headers, values):
            row[h] = v
        data_list.append(row)

df = pd.DataFrame(data_list)
df.to_csv('metric_ablated_loss.csv', index=False)
print(df.head())
print(df.info())