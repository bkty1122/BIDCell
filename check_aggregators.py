import torchjd.aggregation as A
print("Aggregator attributes:")
for x in dir(A):
    if not x.startswith("_"):
        print(x)
