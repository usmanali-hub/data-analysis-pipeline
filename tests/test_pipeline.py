import pandas as pd
import pytest

from datapipe.cleaning import clean_orders, make_messy_orders
from datapipe.pipeline import quality_report, run_pipeline

DETERMINISTIC = ["clean_orders", "order_features", "customer_rfm", "revenue_by_country", "monthly_revenue", "report"]


def test_pipeline_writes_every_output(tmp_path):
    files = run_pipeline(tmp_path, n_orders=120, seed=1)
    assert len(files) == 9
    for path in files.values():
        assert path.exists() and path.stat().st_size > 0


def test_pipeline_is_deterministic(tmp_path):
    a = run_pipeline(tmp_path / "a", n_orders=120, seed=3)
    b = run_pipeline(tmp_path / "b", n_orders=120, seed=3)
    for key in DETERMINISTIC:
        assert a[key].read_bytes() == b[key].read_bytes(), key


def test_different_seeds_give_different_data(tmp_path):
    a = run_pipeline(tmp_path / "a", n_orders=120, seed=1)
    b = run_pipeline(tmp_path / "b", n_orders=120, seed=2)
    assert a["clean_orders"].read_bytes() != b["clean_orders"].read_bytes()


def test_report_numbers_match_the_data(tmp_path):
    files = run_pipeline(tmp_path, n_orders=200, seed=5)
    clean = pd.read_csv(files["clean_orders"])
    by_country = pd.read_csv(files["revenue_by_country"])
    assert by_country["revenue"].sum() == pytest.approx(clean["amount"].sum())
    assert by_country["orders"].sum() == len(clean)
    report = files["report"].read_text(encoding="utf-8")
    assert f"| Rows out | {len(clean)} |" in report


def test_quality_report_counts_known_defects():
    raw = make_messy_orders(200, seed=1)
    q = quality_report(raw, clean_orders(raw))
    assert (q.rows_in, q.rows_out, q.duplicates_removed) == (207, 200, 7)
    assert q.missing_amounts == 6 and q.quantities_imputed == 8 and q.unparsed_dates == 0


def test_pipeline_rejects_nothing_silently_when_output_dir_is_nested(tmp_path):
    files = run_pipeline(tmp_path / "deep" / "er", n_orders=60, seed=0)
    assert files["report"].parent.exists()
