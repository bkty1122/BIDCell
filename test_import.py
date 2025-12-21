
import sys
import os

# Add torchjd to path if needed (assuming it's a sibling of BIDCell)
# The current directory is usually where the script is run. 
# But let's look at where torchjd is relative to BIDCell.
# BIDCell is at D:/2512-BROCK-CODING/BIDCell
# torchjd is at D:/2512-BROCK-CODING/torchjd

sys.path.append("D:/2512-BROCK-CODING/torchjd/src")

try:
    from torchjd.aggregation import UPGrad
    print("UPGrad imported successfully")
except ImportError as e:
    print(f"Failed to import UPGrad: {e}")
