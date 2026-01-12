import inspect
from torchjd.aggregation import UPGrad, Aggregator

print("Aggregator base class methods:")
print([x for x in dir(Aggregator) if not x.startswith("_")])

print("\nUPGrad methods:")
print([x for x in dir(UPGrad) if not x.startswith("_")])

print("\nUPGrad call signature:")
print(inspect.signature(UPGrad.__call__))
