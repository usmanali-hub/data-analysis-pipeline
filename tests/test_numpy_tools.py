import math

import numpy as np
import pytest

from datapipe.numpy_tools import (
    cosine_similarity,
    min_max_scale,
    moving_average,
    one_hot,
    pairwise_euclidean,
    softmax,
    top_k_indices,
    zscore,
)

RNG = np.random.default_rng(42)


# ---- zscore -------------------------------------------------------------
def test_zscore_columns_are_standardized():
    z = zscore(RNG.normal(5, 3, size=(200, 4)))
    assert np.allclose(z.mean(axis=0), 0)
    assert np.allclose(z.std(axis=0), 1)


def test_zscore_constant_column_is_zero_not_nan():
    z = zscore(np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]))
    assert not np.isnan(z).any()
    assert np.all(z[:, 1] == 0)


def test_zscore_axis_one():
    z = zscore(RNG.normal(size=(3, 50)), axis=1)
    assert np.allclose(z.mean(axis=1), 0)


# ---- min_max_scale ------------------------------------------------------
def test_min_max_default_range():
    out = min_max_scale([1, 2, 3, 5])
    assert out.min() == 0 and out.max() == 1
    assert out.tolist() == [0.0, 0.25, 0.5, 1.0]


def test_min_max_custom_range_and_constant():
    assert min_max_scale([2, 4], feature_range=(-1, 1)).tolist() == [-1.0, 1.0]
    assert min_max_scale([7, 7, 7]).tolist() == [0.0, 0.0, 0.0]


def test_min_max_bad_range():
    with pytest.raises(ValueError):
        min_max_scale([1, 2], feature_range=(1, 1))


# ---- moving_average -----------------------------------------------------
def test_moving_average_known_values():
    assert moving_average([1, 2, 3, 4, 5], 3).tolist() == [2.0, 3.0, 4.0]


def test_moving_average_matches_loop():
    x = RNG.normal(size=100)
    expected = [sum(x[i : i + 7]) / 7 for i in range(len(x) - 6)]
    assert np.allclose(moving_average(x, 7), expected)


@pytest.mark.parametrize("window", [0, 6, -1])
def test_moving_average_bad_window(window):
    with pytest.raises(ValueError):
        moving_average([1, 2, 3, 4, 5], window)


def test_moving_average_requires_1d():
    with pytest.raises(ValueError):
        moving_average(np.ones((3, 3)), 2)


# ---- softmax ------------------------------------------------------------
def test_softmax_rows_sum_to_one():
    p = softmax(RNG.normal(size=(5, 8)))
    assert np.allclose(p.sum(axis=1), 1)
    assert (p > 0).all()


def test_softmax_is_stable_for_huge_logits():
    p = softmax([1000.0, 1001.0, 1002.0])
    assert not np.isnan(p).any()
    assert np.allclose(p, softmax([0.0, 1.0, 2.0]))  # shift invariance


def test_softmax_matches_naive_on_small_values():
    x = np.array([0.1, 0.5, -0.3])
    naive = np.exp(x) / np.exp(x).sum()
    assert np.allclose(softmax(x), naive)


# ---- one_hot ------------------------------------------------------------
def test_one_hot_basic():
    assert one_hot([0, 2, 1]).tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]


def test_one_hot_explicit_classes():
    assert one_hot([1], num_classes=4).tolist() == [[0, 1, 0, 0]]


@pytest.mark.parametrize("labels,k", [([-1, 0], None), ([0, 3], 3), ([0.5, 1.0], None)])
def test_one_hot_invalid(labels, k):
    with pytest.raises(ValueError):
        one_hot(labels, num_classes=k)


def test_one_hot_empty():
    assert one_hot([], num_classes=3).shape == (0, 3)


# ---- pairwise_euclidean -------------------------------------------------
def test_pairwise_matches_loop():
    a, b = RNG.normal(size=(6, 3)), RNG.normal(size=(4, 3))
    expected = [[math.dist(p, q) for q in b] for p in a]
    assert np.allclose(pairwise_euclidean(a, b), expected)


def test_pairwise_self_has_zero_diagonal_and_is_symmetric():
    d = pairwise_euclidean(RNG.normal(size=(5, 2)))
    assert np.allclose(np.diag(d), 0)
    assert np.allclose(d, d.T)


def test_pairwise_known_value():
    assert pairwise_euclidean([[0, 0]], [[3, 4]])[0, 0] == 5.0


def test_pairwise_shape_mismatch():
    with pytest.raises(ValueError):
        pairwise_euclidean(np.ones((2, 3)), np.ones((2, 4)))


# ---- cosine_similarity --------------------------------------------------
def test_cosine_known_values():
    sim = cosine_similarity([[1, 0]], [[1, 0], [0, 1], [-1, 0], [1, 1]])
    assert np.allclose(sim[0], [1, 0, -1, 1 / math.sqrt(2)])


def test_cosine_matches_loop_and_is_scale_invariant():
    a, b = RNG.normal(size=(4, 5)), RNG.normal(size=(3, 5))
    expected = [[np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)) for q in b] for p in a]
    assert np.allclose(cosine_similarity(a, b), expected)
    assert np.allclose(cosine_similarity(a * 10, b), expected)


def test_cosine_zero_vector_gives_zero():
    sim = cosine_similarity([[0, 0], [1, 1]], [[1, 0]])
    assert not np.isnan(sim).any()
    assert sim[0, 0] == 0


# ---- top_k_indices ------------------------------------------------------
def test_top_k_orders_best_first():
    assert top_k_indices([0.1, 0.9, 0.5, 0.7], 2).tolist() == [1, 3]


def test_top_k_larger_than_length_returns_all():
    assert top_k_indices([3, 1, 2], 10).tolist() == [0, 2, 1]


def test_top_k_ties_keep_original_order():
    assert top_k_indices([1, 1, 1], 2).tolist() == [0, 1]


@pytest.mark.parametrize("k", [0, -2])
def test_top_k_invalid_k(k):
    with pytest.raises(ValueError):
        top_k_indices([1, 2], k)


def test_retrieval_example_finds_nearest_vector():
    docs = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    query = np.array([[1.0, 0.05]])
    best = top_k_indices(cosine_similarity(query, docs)[0], 1)
    assert best.tolist() == [0]
