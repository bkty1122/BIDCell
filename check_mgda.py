
try:
    from torchjd.aggregation import MGDA
    print("MGDA found")
except ImportError:
    print("MGDA not found")

try:
    from torchjd.aggregation import MGDAUB
    print("MGDAUB found")
except ImportError:
    print("MGDAUB not found")
