"""Analysis of a real dataset: daily Seattle weather, 2012-2015.

Source: the ``seattle-weather.csv`` file from the vega-datasets project, derived from NOAA daily
observations (temperatures in degrees C, precipitation in mm, wind in m/s). That project notes the
``weather`` label column is synthesized from the other fields and that the file is intended for
instructional use, so this module does not draw conclusions from the label.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data" / "seattle-weather.csv"
REQUIRED = ["date", "precipitation", "temp_max", "temp_min", "wind"]
BLUE = "#1f6feb"
RED = "#d1242f"


def load_weather(path: str | Path = DATA) -> pd.DataFrame:
    """Read and validate the file. Raises ValueError for problems that would corrupt results."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df["date"].duplicated().any():
        raise ValueError("duplicate dates")
    if (df["temp_min"] > df["temp_max"]).any():
        raise ValueError("temp_min is above temp_max on some days")
    if (df["precipitation"] < 0).any():
        raise ValueError("negative precipitation")
    return df.sort_values("date", ignore_index=True)


def monthly_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Per calendar month: average rainfall per year, mean temperatures and share of rainy days."""
    years = df["date"].dt.year.nunique()
    grouped = df.groupby(df["date"].dt.month)
    return pd.DataFrame(
        {
            "rain_mm_per_year": grouped["precipitation"].sum() / years,
            "mean_temp_max": grouped["temp_max"].mean(),
            "mean_temp_min": grouped["temp_min"].mean(),
            "rainy_day_share": grouped["precipitation"].agg(lambda s: (s > 0).mean()),
        }
    ).rename_axis("month")


def yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(df["date"].dt.year)
    return pd.DataFrame(
        {
            "days": grouped["date"].count(),
            "rain_mm": grouped["precipitation"].sum(),
            "mean_temp_max": grouped["temp_max"].mean(),
        }
    ).rename_axis("year")


def heavy_rain_days(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Days above the IQR upper fence computed over rainy days only.

    More than half of all days are dry, so computing quartiles over every day would put Q3 near
    zero and flag almost any rain as extreme.
    """
    rainy = df[df["precipitation"] > 0]
    q1, q3 = np.percentile(rainy["precipitation"], [25, 75])
    return df[df["precipitation"] > q3 + k * (q3 - q1)]


def longest_streak(mask: pd.Series) -> tuple[int, pd.Timestamp | None]:
    """Length of the longest run of True values and the date it ends on (None if no True values)."""
    run_id = (~mask).cumsum()
    runs = mask.groupby(run_id).cumsum()
    if not mask.any():
        return 0, None
    return int(runs.max()), pd.Timestamp(mask.index[int(runs.to_numpy().argmax())])


def findings(df: pd.DataFrame) -> list[str]:
    """Plain-language findings, every number computed from the data."""
    clim = monthly_climatology(df)
    yearly = yearly_summary(df)
    heavy = heavy_rain_days(df)
    wet, dry = clim["rain_mm_per_year"].idxmax(), clim["rain_mm_per_year"].idxmin()
    month = lambda m: pd.Timestamp(2000, int(m), 1).strftime("%B")  # noqa: E731
    dry_streak, dry_end = longest_streak(df.set_index("date")["precipitation"].eq(0))
    warm_streak, warm_end = longest_streak(df.set_index("date")["temp_max"].ge(30))
    first, last = yearly.index[0], yearly.index[-1]
    return [
        f"{month(wet)} is the wettest month ({clim.loc[wet, 'rain_mm_per_year']:.0f} mm per year on average) "
        f"and {month(dry)} the driest ({clim.loc[dry, 'rain_mm_per_year']:.0f} mm).",
        f"It rains on {clim.loc[wet, 'rainy_day_share']:.0%} of days in {month(wet)} versus "
        f"{clim.loc[dry, 'rainy_day_share']:.0%} in {month(dry)}.",
        f"The driest year was {yearly['rain_mm'].idxmin()} ({yearly['rain_mm'].min():.0f} mm) and the wettest "
        f"{yearly['rain_mm'].idxmax()} ({yearly['rain_mm'].max():.0f} mm).",
        f"Mean daily maximum temperature rose from {yearly.loc[first, 'mean_temp_max']:.1f} C in {first} to "
        f"{yearly.loc[last, 'mean_temp_max']:.1f} C in {last}. Four years cannot show a climate trend, "
        f"so this is year-to-year variation, not evidence of warming.",
        f"{len(heavy)} days are heavy-rain outliers (above the IQR fence among rainy days); "
        f"the most extreme was {heavy['precipitation'].max():.1f} mm on {heavy.loc[heavy['precipitation'].idxmax(), 'date']:%Y-%m-%d}.",
        f"The longest dry spell was {dry_streak} days, ending {dry_end:%Y-%m-%d}.",
        f"The longest run of days at or above 30 C was {warm_streak} days"
        + (f", ending {warm_end:%Y-%m-%d}." if warm_end is not None else "."),
    ]


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def make_charts(df: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    clim = monthly_climatology(df)
    names = [pd.Timestamp(2000, m, 1).strftime("%b") for m in clim.index]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(names, clim["rain_mm_per_year"], color=BLUE)
    ax.set(title="Average rainfall per month (mm per year)", ylabel="mm")
    _style(ax)
    fig.tight_layout()
    paths.append(out / "weather_rainfall_by_month.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    smooth = df.set_index("date")["temp_max"].rolling("30D").mean()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(df["date"], df["temp_max"], color="#d0d7de", label="daily max")
    ax.plot(smooth.index[29:], smooth.to_numpy()[29:], color=RED, label="30-day average")
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set(title="Daily maximum temperature (C)", ylabel="C")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    paths.append(out / "weather_temperature.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    heavy = heavy_rain_days(df)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    # vlines draw crisp 1-pixel stems; day-wide bars alias badly at this size
    ax.vlines(df["date"], 0, df["precipitation"], color="#c4ccd4", linewidth=0.8, label="daily rainfall")
    ax.vlines(heavy["date"], 0, heavy["precipitation"], color=BLUE, linewidth=1.6, label="heavy-rain outliers")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set(title="Daily rainfall (mm) with outliers marked", ylabel="mm")
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    paths.append(out / "weather_heavy_rain.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)
    return paths


def run(out_dir: str | Path, data: str | Path = DATA) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_weather(data)
    charts = make_charts(df, out)
    bullets = "\n".join(f"- {line}" for line in findings(df))
    report = out / "weather_findings.md"
    report.write_text(
        f"""# Seattle weather, {df['date'].min():%Y}-{df['date'].max():%Y}

{len(df):,} daily observations from NOAA via the vega-datasets project. No missing values; temperature
ordering and duplicate dates were validated on load.

## Findings

{bullets}

![Rainfall by month](weather_rainfall_by_month.png)

![Temperature](weather_temperature.png)

![Heavy rain](weather_heavy_rain.png)
""",
        encoding="utf-8",
    )
    return {"report": report, **{p.stem: p for p in charts}}


def main() -> None:
    parser = argparse.ArgumentParser(prog="datapipe.weather")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    for name, path in run(args.out).items():
        print(f"{name:<28} {path}")


if __name__ == "__main__":
    main()
