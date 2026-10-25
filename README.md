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

### `datapipe.cleaning`

Small pandas steps that each take and return a DataFrame, chained with `.pipe`:

| Function | Purpose |
|---|---|
| `normalize_columns`, `snake_case` | `' Amount ($) '` becomes `amount` |
| `strip_text` | Trim and collapse whitespace in every text column |
| `parse_money` | `'$1,234.50'` becomes `1234.5`; `'N/A'` stays missing instead of becoming zero |
| `parse_mixed_dates` | Tries explicit formats in turn, so day/month order is never guessed per row |
| `standardize_categories` | `'KUWAIT'`, `'kuwait '` and `'Kuwait'` collapse to one label; unknown values are kept |
| `clean_orders` | The full pipeline, tested end to end on `make_messy_orders()` |

### `datapipe.aggregation`

Group, join, pivot and window steps for cleaned orders:

| Function | Purpose |
|---|---|
| `summarize_by` | Orders, revenue and average order value per group, with named aggregation |
| `enrich_with_customers` | Left join with `validate="many_to_one"` so duplicate keys raise instead of multiplying rows; unmatched orders are flagged |
| `monthly_revenue_pivot` | Country by month revenue with totals |
| `add_window_features` | Daily revenue resampled to a full calendar, rolling sum, cumulative sum and share of total |
| `top_n_per_group` | Largest n rows per group, ranked |

Expected values in the tests are computed by hand on a five-row frame, not by re-running the same pandas call.

### `datapipe.features`

Feature engineering that never reads the future:

| Function | Purpose |
|---|---|
| `add_date_features` | Month, weekday, weekend flag, ISO week |
| `add_history_features` | Previous amount, days since the previous order and order number, per customer in date order |
| `customer_rfm` | Recency, frequency and monetary value as of a cut-off date; later orders are ignored |
| `frequency_encode` | Category replaced by its relative frequency |
| `bucket_amount` | Quantile buckets that survive ties and keep missing values missing |

### Notebooks

- [`01_numpy_vectorization.ipynb`](notebooks/01_numpy_vectorization.ipynb): broadcasting, a loop-vs-vectorized distance benchmark, a small embedding-retrieval example and softmax stability, with outputs saved.
- [`02_clean_aggregate_features.ipynb`](notebooks/02_clean_aggregate_features.ipynb): the whole pandas flow on one synthetic table. Messy orders are cleaned, aggregated and joined, then turned into features, with two charts and outputs saved.

## Design notes

- **Edge cases are explicit.** Zero variance, zero-length vectors, empty labels and overflow-prone logits all have defined, tested behavior.
- **Tests compare against loops.** A vectorized function that returns different numbers from the obvious loop is a bug, however fast it is.

## Next

A reproducible pipeline script that runs cleaning, aggregation and features end to end and writes a report with charts.
