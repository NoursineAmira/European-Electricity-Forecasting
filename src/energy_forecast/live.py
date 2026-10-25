"""Daily public forecast: replay the recent past with the backtest code, publish tomorrow.

Every run rebuilds everything from public data, so a failed or skipped day fixes itself.
Running twice a day is safe: a later run replaces the same day's issue (it still only uses
data up to the 09:00 cut; it just has a better chance that all of it has arrived).
"""

from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import backtest, pipeline
from .config import Config
from .experiment import ACI_GAMMA
from .intervals import conformal
from .models import make

HOUR = pd.Timedelta(hours=1)


def _forecast(cfg: Config, country: str, target: str, spec: dict, today: date) -> pd.DataFrame:
    data = pd.read_parquet(cfg.processed_dir / f"{country}.parquet")
    model = make(spec["model"], target, cfg.countries[country], cfg)
    tomorrow = today + timedelta(days=1)
    issue_times = backtest.origins(tomorrow - timedelta(days=cfg.live["replay_days"]), tomorrow, cfg.backtest.issue_hour_utc)

    fc = backtest.run(model, data, issue_times)
    base = "quantiles" if spec["intervals"] == "cqr" else "forecast"
    fc = conformal(fc, cfg.backtest.quantiles, cfg.backtest.warmup_days, ACI_GAMMA, base=base)

    last_known = issue_times[-1] - HOUR
    complete = bool(pd.notna(data[target].get(last_known)))
    out = fc.loc[str(tomorrow)].drop(columns=["origin", "actual"])
    return out.assign(country=country, target=target, issued=str(today), data_complete=complete).reset_index()


def merge_history(history: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append today's issue, replacing an earlier run of the same day."""
    if history.empty:
        return new
    same_issue = history["issued"].isin(new["issued"].unique())
    return pd.concat([history[~same_issue], new], ignore_index=True)


def fill_actuals(history: pd.DataFrame, actuals: dict[tuple[str, str], pd.Series]) -> pd.DataFrame:
    filled = []
    for (country, target), rows in history.groupby(["country", "target"]):
        truth = actuals[country, target].reindex(pd.DatetimeIndex(rows["timestamp"]))
        filled.append(rows.assign(actual=truth.to_numpy()))
    return pd.concat(filled).sort_values(["issued", "country", "target", "timestamp"], ignore_index=True)


def run_live(cfg: Config, out_dir: Path) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    today = now.date()
    cfg = dataclasses.replace(cfg, end=today + timedelta(days=1))
    jobs = [(c, t, spec) for t, spec in cfg.live["publish"].items() for c in spec["countries"]]
    countries = sorted({c for c, _, _ in jobs})

    pipeline.fetch(cfg, countries)
    pipeline.build(cfg, countries)
    with ProcessPoolExecutor() as pool:
        new = pd.concat(pool.map(_forecast, *zip(*[(cfg, c, t, spec, today) for c, t, spec in jobs])))

    data = {c: pd.read_parquet(cfg.processed_dir / f"{c}.parquet") for c in countries}
    actuals = {(c, t): data[c][t] for c, t, _ in jobs}
    recent = pd.concat(
        actuals[c, t].loc[str(today - timedelta(days=2)) :].dropna().rename("actual").reset_index().assign(country=c, target=t)
        for c, t, _ in jobs
    )

    path = out_dir / "history.csv"
    history = pd.read_csv(path, parse_dates=["timestamp"], dtype={"issued": str}) if path.exists() else pd.DataFrame()
    history = fill_actuals(merge_history(history, new), actuals)

    out_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(path, index=False, float_format="%.2f")
    recent.to_csv(out_dir / "recent.csv", index=False, float_format="%.2f")
    status = {
        "issued": str(today),
        "run_at_utc": now.strftime("%Y-%m-%d %H:%M"),
        "data_cut_utc": f"{today} {cfg.backtest.issue_hour_utc - 1:02d}:00",
        "data_complete": {f"{c}/{t}": bool(g["data_complete"].iloc[0]) for (c, t), g in new.groupby(["country", "target"])},
    }
    (out_dir / "status.json").write_text(json.dumps(status, indent=2))
    return new
