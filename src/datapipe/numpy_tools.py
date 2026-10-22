"""Vectorized NumPy building blocks: scaling, smoothing, softmax, distances and similarity.

Every function works on whole arrays (no Python loops) and is tested against a simple
loop-based reference.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def zscore(a: ArrayLike, axis: int = 0, ddof: int = 0) -> FloatArray:
    """Standardize to mean 0 and standard deviation 1 along ``axis``.

    A constant slice has zero standard deviation, so it maps to zeros instead of NaN.
    """
    x = np.asarray(a, dtype=float)
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, ddof=ddof, keepdims=True)
    return (x - mean) / np.where(std == 0, 1.0, std)


def min_max_scale(a: ArrayLike, axis: int = 0, feature_range: tuple[float, float] = (0.0, 1.0)) -> FloatArray:
    """Rescale to ``feature_range`` along ``axis``. A constant slice maps to the lower bound."""
    lo, hi = feature_range
    if lo >= hi:
        raise ValueError("feature_range must satisfy lo < hi")
    x = np.asarray(a, dtype=float)
    mn = x.min(axis=axis, keepdims=True)
    span = x.max(axis=axis, keepdims=True) - mn
    unit = (x - mn) / np.where(span == 0, 1.0, span)
    return unit * (hi - lo) + lo


def moving_average(a: ArrayLike, window: int) -> FloatArray:
    """Simple moving average of a 1-D array. Output length is ``len(a) - window + 1``.

    Uses a cumulative sum, so the cost does not grow with the window size.
    """
    x = np.asarray(a, dtype=float)
    if x.ndim != 1:
        raise ValueError("moving_average expects a 1-D array")
    if not 1 <= window <= len(x):
        raise ValueError("window must be between 1 and len(a)")
    csum = np.cumsum(np.insert(x, 0, 0.0))
    return (csum[window:] - csum[:-window]) / window


def softmax(logits: ArrayLike, axis: int = -1) -> FloatArray:
    """Numerically stable softmax. Subtracting the max first prevents overflow on large logits."""
    x = np.asarray(logits, dtype=float)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def one_hot(labels: ArrayLike, num_classes: int | None = None) -> FloatArray:
    """One-hot encode a 1-D array of non-negative integer labels."""
    y = np.asarray(labels)
    # an empty list becomes a float array in NumPy, so only check the dtype when there is data
    if y.ndim != 1 or (len(y) and not np.issubdtype(y.dtype, np.integer)):
        raise ValueError("labels must be a 1-D integer array")
    y = y.astype(int)
    if len(y) and y.min() < 0:
        raise ValueError("labels must be non-negative")
    k = int(y.max()) + 1 if num_classes is None and len(y) else (num_classes or 0)
    if len(y) and y.max() >= k:
        raise ValueError("num_classes is too small for the given labels")
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def pairwise_euclidean(x: ArrayLike, y: ArrayLike | None = None) -> FloatArray:
    """Distance matrix between rows of ``x`` (n, d) and rows of ``y`` (m, d), via broadcasting.

    Shapes: x[:, None, :] is (n, 1, d) and y[None, :, :] is (1, m, d); their difference
    broadcasts to (n, m, d), and summing over the last axis leaves (n, m).
    """
    a = np.asarray(x, dtype=float)
    b = a if y is None else np.asarray(y, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("expected 2-D arrays with the same number of columns")
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Cosine similarity matrix between rows of ``a`` (n, d) and ``b`` (m, d).

    Rows with zero length have similarity 0 to everything instead of NaN.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("expected 2-D arrays with the same number of columns")
    xn = np.linalg.norm(x, axis=1, keepdims=True)
    yn = np.linalg.norm(y, axis=1, keepdims=True)
    xu = x / np.where(xn == 0, 1.0, xn)
    yu = y / np.where(yn == 0, 1.0, yn)
    return xu @ yu.T


def top_k_indices(scores: ArrayLike, k: int) -> NDArray[np.intp]:
    """Indices of the ``k`` highest scores in a 1-D array, best first. Ties keep original order."""
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1:
        raise ValueError("scores must be 1-D")
    if k < 1:
        raise ValueError("k must be at least 1")
    return np.argsort(-s, kind="stable")[:k]
