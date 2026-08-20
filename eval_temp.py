from src.features import build_feature_matrix
from src.rules_engine import assess

_, _, df = build_feature_matrix()
print(f'Total rows: {len(df)}')
print(f'Mode distributions:')
for mode in ['twf', 'hdf', 'pwf', 'osf', 'rnf']:
    count = int((df[mode] == 1).sum())
    print(f'  {mode.upper()}: {count}')
print(f'Machine failure: {int(df["machine_failure"].sum())}')

# Show a few examples per mode
print('\nSample UDIs per mode:')
for mode in ['twf', 'hdf', 'pwf', 'osf']:
    samples = df[df[mode] == 1]['udi'].tolist()[:3]
    print(f'  {mode.upper()}: {samples}')