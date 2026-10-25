import pandas as pd
import pytest

from datapipe.cleaning import clean_orders, make_messy_orders
from datapipe.features import (
    add_date_features,
    add_history_features,
    bucket_amount,
    customer_rfm,
    frequency_encode,
)


@pytest.fixture
def orders():
    return pd.DataFrame(
        {
            "order_id": [3, 1, 2, 4],
            "customer_name": ["A", "A", "B", "A"],
            "order_date": pd.to_datetime(["2026-01-10", "2026-01-03", "2026-01-04", "2026-01-24"]),
            "amount": [30.0, 10.0, 99.0, 50.0],
        }
    )


def test_date_features():
    out = add_date_features(pd.DataFrame({"order_date": pd.to_datetime(["2026-01-03", "2026-01-05"])}))
    assert out["is_weekend"].tolist() == [True, False]  # Saturday, Monday
    assert out["day_of_week"].tolist() == [5, 0]
    assert out["month"].tolist() == [1, 1]


def test_history_features_are_past_only_and_keep_row_order(orders):
    out = add_history_features(orders)
    assert out["order_id"].tolist() == [3, 1, 2, 4]  # caller's order is preserved
    by_id = out.set_index("order_id")
    assert pd.isna(by_id.loc[1, "prev_amount"]) and pd.isna(by_id.loc[1, "days_since_prev"])
    assert by_id.loc[3, "prev_amount"] == 10.0 and by_id.loc[3, "days_since_prev"] == 7
    assert by_id.loc[4, "prev_amount"] == 30.0 and by_id.loc[4, "days_since_prev"] == 14
    assert by_id.loc[2, "order_number"] == 1 and by_id.loc[4, "order_number"] == 3


def test_rfm_ignores_future_orders(orders):
    rfm = customer_rfm(orders, pd.Timestamp("2026-01-15")).set_index("customer_name")
    assert rfm.loc["A", "frequency"] == 2  # the 2026-01-24 order is in the future
    assert rfm.loc["A", "recency_days"] == 5
    assert rfm.loc["A", "monetary"] == 40.0
    assert rfm.loc["B", "recency_days"] == 11


def test_frequency_encode():
    out = frequency_encode(pd.Series(["x", "x", "y", None]))
    assert out.tolist() == [0.5, 0.5, 0.25, 0.25]


def test_bucket_amount_keeps_missing_and_handles_ties():
    out = bucket_amount(pd.Series([1.0, 2.0, 3.0, 4.0, None]), q=2)
    assert out.iloc[:4].tolist() == [0, 0, 1, 1] and pd.isna(out.iloc[4])
    ties = bucket_amount(pd.Series([5.0, 5.0, None, 5.0]), q=4)  # all ties must not raise or return NaN
    assert ties.tolist()[:2] == [0.0, 0.0] and pd.isna(ties.iloc[2]) and ties.iloc[3] == 0.0


def test_features_on_synthetic_orders():
    clean = clean_orders(make_messy_orders(200, seed=5))
    out = add_history_features(add_date_features(clean))
    assert len(out) == len(clean)
    assert (out["order_number"] >= 1).all()
    assert out.loc[out["order_number"] == 1, "days_since_prev"].isna().all()
    assert (out["days_since_prev"].dropna() >= 0).all()
