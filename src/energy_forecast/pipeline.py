from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .config import Config, Country
from .sources.energy_charts import fetch_power, fetch_price
from .sources.open_meteo import fetch_weather

SOURCES = ("power", "price", "weather")
TARGETS = ("load_mw", "price_eur_mwh")


def quarters(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] into calendar-quarter chunks, both ends inclusive."""
    chunks = []
    cur = start
    while cur <= end:
        q_month = 3 * ((cur.month - 1) // 3) + 1
        next_q = date(cur.year + (q_month == 10), (q_month + 2) % 12 + 1, 1)
        chunk_end = min(next_q - timedelta(days=1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def to_hourly(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Average sub-hourly data to hours on a complete UTC grid; missing hours stay NaN."""
    df = df[~df.index.duplicated()].sort_index()
    grid = pd.date_range(
        pd.Timestamp(start, tz="UTC"),
        pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23),
        freq="h",
        name="timestamp",
    )
    return df.resample("h").mean().reindex(grid)


def clean(source: str, df: pd.DataFrame) -> pd.DataFrame:
    if source == "power" and "load_mw" in df:
        df = df.assign(load_mw=df["load_mw"].where(df["load_mw"] > 0))
    return df


def _raw_path(cfg: Config, country: str, source: str, chunk_start: date) -> Path:
    q = (chunk_start.month - 1) // 3 + 1
    return cfg.raw_dir / country / f"{source}_{chunk_start.year}Q{q}.parquet"


def _download(cfg: Config, c: Country, source: str, start: date, end: date) -> pd.DataFrame:
    if source == "power":
        return fetch_power(c.code, start, end)
    if source == "price":
        return fetch_price(c.bzn, start, end)
    return fetch_weather(c.cities, start, end, cfg.weather_variables)


def fetch(cfg: Config, countries: list[str]) -> None:
    for code in countries:
        c = cfg.countries[code]
        for start, end in quarters(cfg.start, cfg.end):
            for source in SOURCES:
                path = _raw_path(cfg, code, source, start)
                incomplete = end >= cfg.end - timedelta(days=7)
                if path.exists() and not incomplete:
                    continue
                print(f"fetch {code} {source} {start}..{end}")
                df = _download(cfg, c, source, start, end)
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(path)


def build(cfg: Config, countries: list[str]) -> None:
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    for code in countries:
        parts = []
        for source in SOURCES:
            files = sorted((cfg.raw_dir / code).glob(f"{source}_*.parquet"))
            if not files:
                raise FileNotFoundError(f"No raw {source} data for {code}; run fetch first")
            raw = clean(source, pd.concat(pd.read_parquet(f) for f in files))
            parts.append(to_hourly(raw, cfg.start, cfg.end))
        table = pd.concat(parts, axis=1)
        out = cfg.processed_dir / f"{code}.parquet"
        table.to_parquet(out)
        print(f"built {out.name}: {len(table):,} hours, {table.shape[1]} columns")


def _longest_gap_hours(s: pd.Series) -> int:
    missing = s.isna()
    runs = missing.ne(missing.shift()).cumsum()
    return int(missing.groupby(runs).sum().max())


def validate(cfg: Config, countries: list[str]) -> None:
    for code in countries:
        df = pd.read_parquet(cfg.processed_dir / f"{code}.parquet")
        print(f"\n{cfg.countries[code].name} ({code}): {df.index[0]} -> {df.index[-1]}, {len(df):,} hours")
        report = pd.DataFrame(
            {
                "missing_%": df.isna().mean().mul(100).round(2),
                "longest_gap_h": {col: _longest_gap_hours(df[col]) for col in df},
                "min": df.min().round(1),
                "mean": df.mean().round(1),
                "max": df.max().round(1),
            }
        )
        print(report.to_string())
        if not df.index.is_unique or (df.index.to_series().diff().dropna() != pd.Timedelta("1h")).any():
            print("  WARNING: hourly grid is broken")
