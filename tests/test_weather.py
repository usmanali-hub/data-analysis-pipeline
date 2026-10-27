import pandas as pd
import pytest

from datapipe import weather as w


def loop_streak(values, dates):
    best = cur = 0
    end = None
    for v, d in zip(values, dates):
        cur = cur + 1 if v else 0
        if cur > best:
            best, end = cur, d
    return best, end


@pytest.fixture(scope="module")
def df():
    return w.load_weather()


def test_real_file_shape_and_ranges(df):
    assert len(df) == 1461 and df["date"].min() == pd.Timestamp("2012-01-01") and df["date"].max() == pd.Timestamp("2015-12-31")
    assert df[w.REQUIRED].notna().all().all()
    assert (df["temp_min"] <= df["temp_max"]).all() and (df["precipitation"] >= 0).all()


def test_load_validates_bad_files(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("date,precipitation,temp_max,temp_min,wind\n2020-01-01,0,5,9,1\n")
    with pytest.raises(ValueError, match="temp_min"):
        w.load_weather(bad)
    dup = tmp_path / "dup.csv"
    dup.write_text("date,precipitation,temp_max,temp_min,wind\n2020-01-01,0,9,5,1\n2020-01-01,0,9,5,1\n")
    with pytest.raises(ValueError, match="duplicate"):
        w.load_weather(dup)
    cols = tmp_path / "cols.csv"
    cols.write_text("date,rain\n2020-01-01,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        w.load_weather(cols)


def test_longest_streak_matches_a_plain_loop(df):
    for column, mask in (("precipitation", df["precipitation"].eq(0)), ("temp_max", df["temp_max"].ge(30))):
        got = w.longest_streak(pd.Series(mask.to_numpy(), index=df["date"]))
        assert got == loop_streak(mask, df["date"]), column


def test_longest_streak_edge_cases():
    idx = pd.date_range("2020-01-01", periods=4)
    assert w.longest_streak(pd.Series([False] * 4, index=idx)) == (0, None)
    assert w.longest_streak(pd.Series([True] * 4, index=idx)) == (4, idx[-1])
    assert w.longest_streak(pd.Series([True, False, True, True], index=idx)) == (2, idx[-1])


def test_monthly_climatology_matches_hand_totals(df):
    clim = w.monthly_climatology(df)
    nov = df[df["date"].dt.month == 11]["precipitation"].sum() / 4
    assert clim.loc[11, "rain_mm_per_year"] == pytest.approx(nov)
    assert clim["rain_mm_per_year"].idxmax() == 11 and clim["rain_mm_per_year"].idxmin() == 7
    assert clim["rainy_day_share"].between(0, 1).all()


def test_heavy_rain_uses_rainy_days_only():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=12), "precipitation": [0.0] * 6 + [1, 1, 2, 2, 3, 40.0]})
    heavy = w.heavy_rain_days(df)
    assert heavy["precipitation"].tolist() == [40.0]  # the 1-3 mm days are not flagged, though most days are dry


def test_findings_quote_numbers_that_match_the_data(df):
    text = " ".join(w.findings(df))
    assert "November is the wettest month (161 mm" in text and "July the driest (12 mm" in text
    assert "48 days" in text and "Four years cannot show a climate trend" in text


def test_run_writes_report_and_charts(tmp_path):
    files = w.run(tmp_path)
    assert len(files) == 4 and all(p.stat().st_size > 500 for p in files.values())
    assert "## Findings" in files["report"].read_text(encoding="utf-8")
