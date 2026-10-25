"""Feature engineering for cleaned orders.

Every function returns a new DataFrame or Series and never mutates its input. Features that look
at the past (lags, days since the previous order) are computed per customer in date order, so a
row never sees information from its own future.
"""

from __future__ import annotations

import pandas as pd


def add_date_features(df: pd.DataFrame, date_col: str = "order_date") -> pd.DataFrame:
    dates = df[date_col]
    return df.assign(
        month=dates.dt.month,
        day_of_week=dates.dt.dayofweek,  # Monday = 0
        is_weekend=dates.dt.dayofweek >= 5,
        week_of_year=dates.dt.isocalendar().week.astype(int),
    )


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Previous amount and days since the customer's previous order (NaN for a first order)."""
    ordered = df.sort_values(["customer_name", "order_date", "order_id"])
    grouped = ordered.groupby("customer_name")
    out = ordered.assign(
        prev_amount=grouped["amount"].shift(1),
        days_since_prev=(ordered["order_date"] - grouped["order_date"].shift(1)).dt.days,
        order_number=grouped.cumcount() + 1,
    )
    return out.loc[df.index]  # restore the caller's row order


def customer_rfm(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Recency, frequency and monetary value per customer, measured at ``as_of``.

    Orders after ``as_of`` are ignored, which is how a model would see the data on that day.
    """
    seen = df[df["order_date"] <= as_of]
    rfm = seen.groupby("customer_name").agg(
        recency_days=("order_date", lambda s: (as_of - s.max()).days),
        frequency=("order_id", "count"),
        monetary=("amount", "sum"),
    )
    return rfm.reset_index()


def frequency_encode(series: pd.Series) -> pd.Series:
    """Replace each category by its relative frequency (a leakage-free encoding for rare labels)."""
    return series.map(series.value_counts(normalize=True, dropna=False))


def bucket_amount(amounts: pd.Series, q: int = 4) -> pd.Series:
    """Quantile buckets labelled 0..q-1; missing amounts stay missing.

    ``duplicates="drop"`` keeps qcut from failing when many values tie on a bin edge. If every
    value is identical there is nothing to split, so all non-missing values go in bucket 0 (qcut
    would otherwise return only NaN).
    """
    if amounts.dropna().nunique() <= 1:
        return pd.Series(0.0, index=amounts.index).where(amounts.notna())
    return pd.qcut(amounts, q=q, labels=False, duplicates="drop")
