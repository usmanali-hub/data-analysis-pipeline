"""End-to-end pipeline: messy orders in, clean tables, features, charts and a report out.

Everything is deterministic for a given seed, so two runs write identical CSV and Markdown files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed in CI
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .aggregation import (  # noqa: E402
    add_window_features,
    enrich_with_customers,
    make_customers,
    monthly_revenue_pivot,
    summarize_by,
)
from .cleaning import clean_orders, make_messy_orders  # noqa: E402
from .features import add_date_features, add_history_features, bucket_amount, customer_rfm  # noqa: E402

BLUE = "#1f6feb"
GREY = "#8b949e"


@dataclass(frozen=True)
class QualityReport:
    rows_in: int
    rows_out: int
    duplicates_removed: int
    missing_amounts: int
    unparsed_dates: int
    quantities_imputed: int
    unknown_countries: int


def quality_report(raw: pd.DataFrame, clean: pd.DataFrame) -> QualityReport:
    return QualityReport(
        rows_in=len(raw),
        rows_out=len(clean),
        duplicates_removed=int(raw.duplicated().sum()),
        missing_amounts=int(clean["amount"].isna().sum()),
        unparsed_dates=int(clean["order_date"].isna().sum()),
        quantities_imputed=int(clean["quantity_missing"].sum()),
        unknown_countries=int(clean["country"].isna().sum()),
    )


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def chart_revenue_trend(daily: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(daily.index, daily["revenue"], color="#d0d7de", label="daily")
    full = daily.iloc[6:]  # the first 6 days have a partial window, which would understate the average
    ax.plot(full.index, full["rolling_revenue"] / 7, color=BLUE, label="7-day average")
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set(title="Daily revenue and 7-day average", ylabel="revenue")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, path)


def chart_revenue_by_country(summary: pd.DataFrame, path: Path) -> None:
    data = summary.sort_values("revenue")
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.barh(data["country"], data["revenue"], color=BLUE)
    ax.set(title="Revenue by country", xlabel="revenue")
    _style(ax)
    _save(fig, path)


def chart_value_buckets(rfm: pd.DataFrame, path: Path) -> None:
    means = rfm.groupby("value_bucket")["monetary"].mean()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(means.index.astype(int).astype(str), means.values, color=BLUE)
    ax.set(title="Mean spend by customer value bucket", xlabel="bucket (0 = lowest)", ylabel="mean spend")
    _style(ax)
    _save(fig, path)


def run_pipeline(out_dir: str | Path, n_orders: int = 500, seed: int = 42) -> dict[str, Path]:
    """Run every stage and write outputs into ``out_dir``. Returns the files written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = make_messy_orders(n_orders, seed=seed)
    clean = clean_orders(raw)
    quality = quality_report(raw, clean)

    customers = make_customers(sorted(clean["customer_name"].unique()))
    enriched = enrich_with_customers(clean, customers)
    by_country = summarize_by(enriched.assign(country=enriched["country"].fillna("(unknown)")), "country")
    by_segment = summarize_by(enriched, "segment")
    monthly = monthly_revenue_pivot(clean)
    daily = add_window_features(clean, window_days=7)

    features = add_history_features(add_date_features(clean))
    as_of = clean["order_date"].quantile(0.8).normalize()
    rfm = customer_rfm(clean, as_of)
    rfm["value_bucket"] = bucket_amount(rfm["monetary"], q=4)

    files = {
        "clean_orders": out / "clean_orders.csv",
        "order_features": out / "order_features.csv",
        "customer_rfm": out / "customer_rfm.csv",
        "revenue_by_country": out / "revenue_by_country.csv",
        "monthly_revenue": out / "monthly_revenue.csv",
        "report": out / "report.md",
        "chart_trend": out / "revenue_trend.png",
        "chart_country": out / "revenue_by_country.png",
        "chart_buckets": out / "value_buckets.png",
    }
    clean.to_csv(files["clean_orders"], index=False)
    features.to_csv(files["order_features"], index=False)
    rfm.to_csv(files["customer_rfm"], index=False)
    by_country.to_csv(files["revenue_by_country"], index=False)
    monthly.to_csv(files["monthly_revenue"])
    chart_revenue_trend(daily, files["chart_trend"])
    chart_revenue_by_country(by_country, files["chart_country"])
    chart_value_buckets(rfm, files["chart_buckets"])
    files["report"].write_text(
        render_report(quality, by_country, by_segment, as_of, seed), encoding="utf-8"
    )
    return files


def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = [
        "| " + " | ".join(f"{v:,.2f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, rule, *rows])


def render_report(
    quality: QualityReport,
    by_country: pd.DataFrame,
    by_segment: pd.DataFrame,
    as_of: pd.Timestamp,
    seed: int,
) -> str:
    return f"""# Orders pipeline report

Synthetic data, seed {seed}. Customer features are measured as of {as_of.date()}.

## Data quality

| Check | Count |
|---|---|
| Rows in | {quality.rows_in} |
| Rows out | {quality.rows_out} |
| Exact duplicates removed | {quality.duplicates_removed} |
| Amounts missing after parsing (kept as missing, not zero) | {quality.missing_amounts} |
| Dates that could not be parsed | {quality.unparsed_dates} |
| Quantities imputed with the median (flagged) | {quality.quantities_imputed} |
| Orders with no country | {quality.unknown_countries} |

## Revenue by country

{_md_table(by_country.round(2))}

## Revenue by customer segment

{_md_table(by_segment.round(2))}

![Revenue trend](revenue_trend.png)

![Revenue by country](revenue_by_country.png)

![Spend by value bucket](value_buckets.png)
"""
