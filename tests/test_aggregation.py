import pandas as pd
import pytest

from datapipe.aggregation import (
    add_window_features,
    enrich_with_customers,
    make_customers,
    monthly_revenue_pivot,
    summarize_by,
    top_n_per_group,
)
from datapipe.cleaning import clean_orders, make_messy_orders


@pytest.fixture
def orders():
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "customer_name": ["A", "A", "B", "C", "Z"],
            "country": ["Kuwait", "Kuwait", "UAE", "UAE", "UAE"],
            "order_date": pd.to_datetime(
                ["2026-01-01", "2026-01-03", "2026-01-03", "2026-02-10", "2026-02-10"]
            ),
            "amount": [100.0, 50.0, 30.0, None, 20.0],
            "quantity": [1, 2, 1, 1, 1],
        }
    )


def test_summarize_by_hand_computed(orders):
    out = summarize_by(orders, "country").set_index("country")
    assert out.loc["Kuwait", "revenue"] == 150.0
    assert out.loc["Kuwait", "avg_order_value"] == 75.0
    assert out.loc["UAE", "orders"] == 3
    assert out.loc["UAE", "priced_orders"] == 2  # the missing amount is not counted as a price
    assert out.loc["UAE", "avg_order_value"] == 25.0
    assert out.index[0] == "Kuwait"  # sorted by revenue, descending


def test_enrich_keeps_unmatched_orders_and_flags_them(orders):
    customers = pd.DataFrame({"customer_name": ["A", "B", "C"], "segment": ["retail", "wholesale", "retail"]})
    out = enrich_with_customers(orders, customers)
    assert len(out) == len(orders)  # a left join must not drop or multiply rows
    z = out[out["customer_name"] == "Z"].iloc[0]
    assert z["segment"] == "unknown" and not z["customer_matched"]
    assert out["customer_matched"].sum() == 4


def test_enrich_rejects_duplicate_customer_keys(orders):
    dup = pd.DataFrame({"customer_name": ["A", "A"], "segment": ["x", "y"]})
    with pytest.raises(pd.errors.MergeError):
        enrich_with_customers(orders, dup)


def test_monthly_pivot_totals(orders):
    pivot = monthly_revenue_pivot(orders)
    assert pivot.loc["Kuwait", "2026-01"] == 150.0
    assert pivot.loc["UAE", "2026-02"] == 20.0
    assert pivot.loc["Total", "Total"] == 200.0


def test_window_features_fill_empty_days(orders):
    out = add_window_features(orders, window_days=3)
    assert out.index.is_monotonic_increasing and len(out) == 41  # Jan 1 through Feb 10
    assert out.loc["2026-01-02", "revenue"] == 0.0
    assert out.loc["2026-01-03", "rolling_revenue"] == 180.0  # Jan 1 + Jan 3 inside the 3-day window
    assert out.loc["2026-01-04", "rolling_revenue"] == 80.0  # Jan 1 has dropped out
    assert out["cumulative_revenue"].iloc[-1] == 200.0
    assert out["share_of_total"].sum() == pytest.approx(1.0)


def test_top_n_per_group(orders):
    out = top_n_per_group(orders.dropna(), "country", "amount", n=1)
    assert out["order_id"].tolist() == [1, 3]
    assert out["rank"].tolist() == [1, 1]


def test_pipeline_on_synthetic_orders():
    clean = clean_orders(make_messy_orders(200, seed=2))
    customers = make_customers(sorted(clean["customer_name"].unique()))
    enriched = enrich_with_customers(clean, customers)
    assert len(enriched) == len(clean) and enriched["customer_matched"].all()
    by_segment = summarize_by(enriched, "segment")
    assert by_segment["orders"].sum() == len(clean)
    assert by_segment["revenue"].sum() == pytest.approx(clean["amount"].sum())
