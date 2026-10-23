import pandas as pd
import pytest

from datapipe.cleaning import (
    clean_orders,
    make_messy_orders,
    normalize_columns,
    parse_mixed_dates,
    parse_money,
    snake_case,
    standardize_categories,
    strip_text,
)


@pytest.mark.parametrize(
    "raw,expected",
    [(" Order ID ", "order_id"), ("Amount ($)", "amount"), ("Customer  Name", "customer_name"), ("A-B", "a_b")],
)
def test_snake_case(raw, expected):
    assert snake_case(raw) == expected


def test_normalize_columns_does_not_mutate_input():
    df = pd.DataFrame({" A B ": [1]})
    out = normalize_columns(df)
    assert list(out.columns) == ["a_b"]
    assert list(df.columns) == [" A B "]


def test_strip_text_trims_and_collapses():
    df = pd.DataFrame({"name": ["  ana   b  ", "c"], "n": [1, 2]})
    assert strip_text(df)["name"].tolist() == ["ana b", "c"]


def test_parse_money_handles_symbols_and_junk():
    s = pd.Series(["$1,234.50", "99", "N/A", "", None])
    out = parse_money(s)
    assert out.iloc[:2].tolist() == [1234.5, 99.0]
    assert out.iloc[2:].isna().all()


def test_parse_mixed_dates_tries_each_format():
    s = pd.Series(["2026-03-05", "05/03/2026", "Mar 05, 2026", "garbage"])
    out = parse_mixed_dates(s, ("%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"))
    assert (out.iloc[:3] == pd.Timestamp("2026-03-05")).all()
    assert pd.isna(out.iloc[3])


def test_standardize_categories_keeps_unknown():
    s = pd.Series(["KUWAIT", "U.A.E", "Oman"])
    out = standardize_categories(s, {"kuwait": "Kuwait", "uae": "UAE"})
    assert out.tolist() == ["Kuwait", "UAE", "Oman"]


def test_clean_orders_end_to_end():
    raw = make_messy_orders(200, seed=1)
    clean = clean_orders(raw)
    assert len(clean) == 200  # 7 exact duplicates removed
    assert "order_id" in clean.columns
    assert pd.api.types.is_datetime64_any_dtype(clean["order_date"])
    assert clean["order_date"].notna().all()
    assert clean["amount"].isna().sum() == 6  # the 'N/A' rows stay missing, not zero
    assert set(clean["country"].dropna()) <= {"Kuwait", "Pakistan", "UAE"}
    assert clean["quantity"].dtype.kind == "i"
    assert clean["quantity_missing"].sum() == 8


def test_clean_orders_is_deterministic():
    a = clean_orders(make_messy_orders(50, seed=3))
    b = clean_orders(make_messy_orders(50, seed=3))
    pd.testing.assert_frame_equal(a, b)
