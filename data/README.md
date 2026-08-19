# Data

## `ai4i2020.csv` - AI4I 2020 Predictive Maintenance Dataset

Committed to git on purpose (510 KB). Every dev trains on a byte-identical file,
so metrics are comparable across machines and nobody loses 20 minutes to a
Kaggle login at hour 30.

**Integrity check** - run this before you trust any number you produce:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/ai4i2020.csv')
assert df.shape == (10000, 14), df.shape
assert df['Machine failure'].sum() == 339
print({c: int(df[c].sum()) for c in ['Machine failure','TWF','HDF','PWF','OSF','RNF']})
"
```

Expected output:
`{'Machine failure': 339, 'TWF': 46, 'HDF': 115, 'PWF': 95, 'OSF': 98, 'RNF': 19}`

MD5 (after BOM strip): see PROGRESS.md.

## Provenance

- Original: S. Matzka, *Explainable Artificial Intelligence for Predictive
  Maintenance Applications*, 2020. UCI ML Repository id **601**.
- Kaggle mirror: `stephanmatzka/predictive-maintenance-dataset-ai4i-2020`
- Programmatic fallback if this file ever goes missing:
  ```python
  from ucimlrepo import fetch_ucirepo
  d = fetch_ucirepo(id=601)
  ```

## Structure

10,000 rows, 14 columns, 339 failures (3.39% positive rate). Synthetic but
physically grounded. **No timestamps and no machine IDs** - rows are independent
snapshots, not a time series. `UDI` is used purely as a replay index to simulate
a live stream in the demo. We say this out loud in the pitch rather than letting
a judge find it.

## The leakage trap

`Machine failure = TWF OR HDF OR PWF OR OSF OR RNF`.

The five mode flags **are** the target. Using them as input features yields
~99.9% accuracy and destroys credibility the moment a judge asks. The drop list
is enforced in code by `src/features.py::assert_no_leakage()` - do not work
around it.
