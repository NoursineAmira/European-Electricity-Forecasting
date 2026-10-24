from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from itertools import product
from pathlib import Path

import pandas as pd

from . import backtest
from .config import Config
from .metrics import score
from .models import MODELS

REFERENCE = "naive_weekly"


def _path(cfg: Config, country: str, target: str, model: str) -> Path:
    return cfg.backtests_dir / country / target / f"{model}.parquet"


def _run_one(cfg: Config, country: str, target: str, model_name: str) -> str:
    bt = cfg.backtest
    series = pd.read_parquet(cfg.processed_dir / f"{country}.parquet")[target]
    first_day = bt.test_start - timedelta(days=bt.warmup_days)
    issue_times = backtest.origins(first_day, bt.test_end, bt.issue_hour_utc)

    fc = backtest.run(MODELS[model_name](), series, issue_times)
    fc = backtest.add_quantiles(fc, bt.quantiles, bt.warmup_days)

    out = _path(cfg, country, target, model_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out)
    return f"{country} {target} {model_name}: {len(fc):,} forecast hours"


def run_backtests(cfg: Config, countries: list[str], targets: list[str], models: list[str]) -> None:
    jobs = list(product(countries, targets, models))
    with ProcessPoolExecutor() as pool:
        for msg in pool.map(_run_one, [cfg] * len(jobs), *zip(*jobs)):
            print(msg)


def _load_test_period(cfg: Config, country: str, target: str) -> dict[str, pd.DataFrame]:
    """All models' forecasts for one series, restricted to test hours every model could score."""
    bt = cfg.backtest
    start = pd.Timestamp(bt.test_start, tz="UTC")
    end = pd.Timestamp(bt.test_end, tz="UTC") + pd.Timedelta(hours=23)
    frames = {p.stem: pd.read_parquet(p).loc[start:end] for p in sorted((cfg.backtests_dir / country / target).glob("*.parquet"))}
    common = None
    for df in frames.values():
        ok = df.dropna(subset=["actual", "forecast"]).index
        common = ok if common is None else common.intersection(ok)
    return {name: df.loc[common] for name, df in frames.items()}


def evaluate(cfg: Config, countries: list[str], targets: list[str]) -> pd.DataFrame:
    rows, by_hour, by_month = [], [], []
    for country, target in product(countries, targets):
        frames = _load_test_period(cfg, country, target)
        if not frames:
            continue
        for name, df in frames.items():
            s = score(df, cfg.backtest.quantiles, with_mape=target == "load_mw")
            rows.append({"country": country, "target": target, "model": name, **s})
            abs_err = (df["actual"] - df["forecast"]).abs()
            key = {"country": country, "target": target, "model": name}
            by_hour.append(abs_err.groupby(df.index.hour).mean().rename_axis("hour").reset_index(name="mae").assign(**key))
            by_month.append(abs_err.groupby(df.index.strftime("%Y-%m")).mean().rename_axis("month").reset_index(name="mae").assign(**key))

    metrics = pd.DataFrame(rows)
    ref = metrics[metrics.model == REFERENCE].set_index(["country", "target"])["mae"]
    metrics["skill_%"] = (1 - metrics["mae"] / ref.reindex(pd.MultiIndex.from_frame(metrics[["country", "target"]])).to_numpy()) * 100

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    metrics.round(2).to_csv(cfg.results_dir / "metrics.csv", index=False)
    pd.concat(by_hour).round(1).to_csv(cfg.results_dir / "mae_by_hour.csv", index=False)
    pd.concat(by_month).round(1).to_csv(cfg.results_dir / "mae_by_month.csv", index=False)
    return metrics
