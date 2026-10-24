"""Aggregation, joins, pivots and window calculations on cleaned orders.

Expects the columns produced by ``datapipe.cleaning.clean_orders``:
``order_id, customer_name, country, order_date, amount, quantity``.
"""

from __future__ import annotations

import pandas as pd


def summarize_by(df: pd.DataFrame, keys: str | list[str]) -> pd.DataFrame:
    """Orders, revenue and average order value per group.

    Named aggregation keeps output column names explicit. Missing amounts are skipped by sum and
    mean, so ``orders`` counts every order while ``priced_orders`` counts those with an amount.
    """
    return (
        df.groupby(keys, dropna=False)
        .agg(
            orders=("order_id", "count"),
            priced_orders=("amount", "count"),
            revenue=("amount", "sum"),
            avg_order_value=("amount", "mean"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False, ignore_index=True)
    )


def enrich_with_customers(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Left-join customer attributes onto orders.

    ``validate="many_to_one"`` makes pandas raise if the customer table has duplicate keys, which
    would silently multiply order rows. Orders with no matching customer are kept and labelled.
    """
    merged = orders.merge(
        customers, on="customer_name", how="left", validate="many_to_one", indicator=True
    )
    merged["segment"] = merged["segment"].fillna("unknown")
    return merged.assign(customer_matched=merged.pop("_merge").eq("both"))


def monthly_revenue_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue with countries as rows and calendar months as columns, plus a Total row and column."""
    month = df["order_date"].dt.to_period("M").astype(str)
    return df.assign(month=month).pivot_table(
        index="country",
        columns="month",
        values="amount",
        aggfunc="sum",
        fill_value=0.0,
        margins=True,
        margins_name="Total",
    )


def add_window_features(df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Daily revenue with a time-based rolling sum, cumulative total and share of the grand total.

    Resampling to a complete daily index first makes days without orders count as zero, so the
    rolling window always spans ``window_days`` calendar days instead of N rows.
    """
    daily = df.set_index("order_date")["amount"].resample("D").sum()
    out = daily.to_frame("revenue")
    out["rolling_revenue"] = out["revenue"].rolling(f"{window_days}D").sum()
    out["cumulative_revenue"] = out["revenue"].cumsum()
    total = out["revenue"].sum()
    out["share_of_total"] = out["revenue"] / total if total else 0.0
    return out


def top_n_per_group(df: pd.DataFrame, group: str, by: str, n: int = 3) -> pd.DataFrame:
    """The n largest rows of ``by`` inside each ``group``, ranked."""
    ranked = df.assign(rank=df.groupby(group)[by].rank(method="first", ascending=False))
    return (
        ranked[ranked["rank"] <= n]
        .sort_values([group, "rank"])
        .astype({"rank": int})
        .reset_index(drop=True)
    )


def make_customers(names: list[str], seed: int = 0) -> pd.DataFrame:
    """Small customer dimension table with a segment per customer."""
    segments = ["retail", "wholesale", "enterprise"]
    return pd.DataFrame(
        {"customer_name": names, "segment": [segments[(i + seed) % 3] for i in range(len(names))]}
    )
