# data-analysis-pipeline

A reproducible data analysis toolkit in Python, built step by step: a tested NumPy foundation first,
then pandas cleaning, aggregation and feature engineering, finishing in one runnable pipeline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## What's inside

### `datapipe.numpy_tools`

Vectorized building blocks, each tested against a plain-Python loop reference:

| Function | Purpose |
|---|---|
| `zscore`, `min_max_scale` | Feature scaling along any axis; constant columns map to zeros instead of NaN |
| `moving_average` | Smoothing via cumulative sum, cost independent of window size |
| `softmax` | Numerically stable (subtracts the max), safe for logits in the thousands |
| `one_hot` | Label encoding with validation |
| `pairwise_euclidean` | Distance matrix using broadcasting: (n,1,d) minus (1,m,d) gives (n,m,d) |
| `cosine_similarity`, `top_k_indices` | The core of embedding search: score every vector, take the best k |

```python
import numpy as np
from datapipe.numpy_tools import cosine_similarity, top_k_indices

docs = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
query = np.array([[1.0, 0.05]])
print(top_k_indices(cosine_similarity(query, docs)[0], k=2))   # [0 2]
```

## Design notes

- **Edge cases are explicit.** Zero variance, zero-length vectors, empty labels and overflow-prone logits all have defined, tested behavior.
- **Tests compare against loops.** A vectorized function that returns different numbers from the obvious loop is a bug, however fast it is.

## Next

pandas cleaning, aggregation and feature engineering on a messy synthetic dataset, combined into one reproducible pipeline script.
