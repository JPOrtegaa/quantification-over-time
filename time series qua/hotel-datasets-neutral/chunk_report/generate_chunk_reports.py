#!/usr/bin/env python3
"""Build chunk-size reports (day / week / month) for each hotel CSV in the parent folder."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

NEUTRAL_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent


def load_filtered(path: Path) -> pd.DataFrame:
    """Match experiment loader: drop rating, valid class label, UTC dates."""
    df = pd.read_csv(path)
    df = df.drop(columns=["rating"], errors="ignore")
    if "class" not in df.columns:
        raise ValueError(f"{path.name}: missing 'class'")
    df = df.rename(columns={"class": "label"})
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([-1, 0, 1])]
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date", "label"])
    df["utc_day"] = df["date"].dt.normalize()
    return df


def daily_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("utc_day", sort=True).size().rename("n_reviews").reset_index()
    g["utc_day"] = g["utc_day"].dt.strftime("%Y-%m-%d")
    return g


def weekly_table(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("utc_day", sort=True).size().rename("n_reviews")
    daily_by_week = daily.reset_index()
    daily_by_week["iso_week"] = daily_by_week["utc_day"].dt.strftime("%G-W%V")

    out = daily_by_week.groupby("iso_week", sort=True).agg(
        n_reviews=("n_reviews", "sum"),
        n_days_with_reviews=("n_reviews", "count"),
        min_daily_n=("n_reviews", "min"),
        max_daily_n=("n_reviews", "max"),
        median_daily_n=("n_reviews", "median"),
        mean_daily_n=("n_reviews", "mean"),
    ).reset_index()
    # ISO week start (Monday) UTC for readability
    week_starts = []
    for wk in out["iso_week"]:
        year, _, week = wk.partition("-W")
        week_i = int(week)
        y_i = int(year)
        week_start = pd.Timestamp.fromisocalendar(y_i, week_i, 1).tz_localize("UTC")
        week_starts.append(week_start.strftime("%Y-%m-%d"))
    out.insert(1, "week_start_utc_monday", week_starts)
    return out


def monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("utc_day", sort=True).size().rename("n_reviews")
    daily_by_m = daily.reset_index()
    daily_by_m["year_month"] = daily_by_m["utc_day"].dt.strftime("%Y-%m")

    return daily_by_m.groupby("year_month", sort=True).agg(
        n_reviews=("n_reviews", "sum"),
        n_days_with_reviews=("n_reviews", "count"),
        min_daily_n=("n_reviews", "min"),
        max_daily_n=("n_reviews", "max"),
        median_daily_n=("n_reviews", "median"),
        mean_daily_n=("n_reviews", "mean"),
    ).reset_index()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csvs = sorted(NEUTRAL_DIR.glob("hotel*.csv"))
    if not csvs:
        raise SystemExit(f"No hotel*.csv under {NEUTRAL_DIR}")

    summary_rows = []
    for path in csvs:
        stem = path.stem
        df = load_filtered(path)
        d = daily_table(df)
        w = weekly_table(df)
        m = monthly_table(df)

        d.to_csv(OUT_DIR / f"{stem}_chunks_by_day.csv", index=False)
        w.to_csv(OUT_DIR / f"{stem}_chunks_by_week.csv", index=False)
        m.to_csv(OUT_DIR / f"{stem}_chunks_by_month.csv", index=False)

        summary_rows.append(
            {
                "hotel": stem,
                "n_reviews": len(df),
                "n_daily_chunks": len(d),
                "n_weekly_chunks": len(w),
                "n_monthly_chunks": len(m),
                "median_daily_chunk_size": float(d["n_reviews"].median()),
                "median_weekly_chunk_size": float(w["n_reviews"].median()),
                "median_monthly_chunk_size": float(m["n_reviews"].median()),
            }
        )

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary_all_hotels.csv", index=False)
    print(f"Wrote reports for {len(csvs)} hotel(s) under {OUT_DIR}")


if __name__ == "__main__":
    main()
