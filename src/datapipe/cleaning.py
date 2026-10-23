"""pandas cleaning steps for messy tabular data.

Each step is a small function that takes a DataFrame and returns a new one, so steps can be
tested alone and chained with ``DataFrame.pipe``. Nothing here mutates its input.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def make_messy_orders(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic orders table with the usual real-world problems baked in."""
    rng = np.random.default_rng(seed)
    countries = ["Kuwait", "kuwait ", "KUWAIT", "Pakistan", "pakistan", "UAE", "U.A.E", None]
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 120, n), unit="D")
    styles = rng.integers(0, 3, n)
    order_date = [
        d.strftime(["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"][s]) for d, s in zip(dates, styles)
    ]
    amounts = rng.uniform(5, 500, n).round(2)
    amount = [f"${a:,.2f}" if i % 3 else f"{a}" for i, a in enumerate(amounts)]
    df = pd.DataFrame(
        {
            " Order ID ": np.arange(1000, 1000 + n),
            "Customer Name": [f"  customer {i % 40}  " for i in range(n)],
            "Country": rng.choice(countries, n),
            "Order Date": order_date,
            "Amount ($)": amount,
            "Quantity": rng.integers(1, 6, n).astype(float),
        }
    )
    df.loc[rng.choice(n, 8, replace=False), "Quantity"] = np.nan
    df.loc[rng.choice(n, 6, replace=False), "Amount ($)"] = "N/A"
    return pd.concat([df, df.iloc[:7]], ignore_index=True)  # exact duplicate rows


def snake_case(name: str) -> str:
    """' Amount ($) ' -> 'amount'."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_")
    return name.lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=snake_case)


def strip_text(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace in every text column; collapse inner runs of spaces."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].str.strip().str.replace(r"\s+", " ", regex=True)
    return out


def parse_money(series: pd.Series) -> pd.Series:
    """'$1,234.50' -> 1234.5; anything unparseable ('N/A', '') -> NaN."""
    cleaned = series.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_mixed_dates(series: pd.Series, formats: tuple[str, ...]) -> pd.Series:
    """Try each format in turn; values matching none become NaT.

    Trying explicit formats avoids pandas guessing day/month order differently per row.
    """
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in formats:
        pending = result.isna()
        result[pending] = pd.to_datetime(series[pending], format=fmt, errors="coerce")
    return result


def standardize_categories(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Map lower-cased, dot-stripped values through ``mapping``; unknown values are kept as-is."""
    key = series.str.lower().str.replace(".", "", regex=False)
    return key.map(mapping).fillna(series)


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for the synthetic orders table."""
    return (
        df.pipe(normalize_columns)
        .pipe(strip_text)
        .drop_duplicates()
        .assign(
            amount=lambda d: parse_money(d["amount"]),
            order_date=lambda d: parse_mixed_dates(
                d["order_date"], ("%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y")
            ),
            country=lambda d: standardize_categories(
                d["country"], {"kuwait": "Kuwait", "pakistan": "Pakistan", "uae": "UAE"}
            ),
            quantity_missing=lambda d: d["quantity"].isna(),
        )
        .assign(quantity=lambda d: d["quantity"].fillna(d["quantity"].median()).astype(int))
        .reset_index(drop=True)
    )
