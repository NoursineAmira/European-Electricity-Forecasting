from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest
from .config import Config
from .features import calendar
from .intervals import conformal, winkler
from .metrics import pinball, score
from .models import make
from .significance import block_bootstrap_skill, daily_mae, diebold_mariano, holm

REFERENCE = "naive_weekly"
ACI_GAMMA = 0.005  # Gibbs & Candès (2021); not tuned, see docs/05
GAMMA_SENSITIVITY = (0.001, 0.005, 0.02)
DM_COMPARISONS = ("naive_weekly", "weekly_mean", "mstl", "lgbm_noweather", "lgbm_oracle")


@dataclass(frozen=True)
class Job:
    country: str
    target: str
    model: str
    first_day: date
    last_day: date
    out: Path
    overrides: dict = field(default_factory=dict)


def _run(job: Job, cfg: Config) -> str:
    bt = cfg.backtest
    data = pd.read_parquet(cfg.processed_dir / f"{job.country}.parquet")
    model = make(job.model, job.target, cfg.countries[job.country], cfg, job.overrides)

    issue_times = backtest.origins(job.first_day, job.last_day, bt.issue_hour_utc)
    fc = backtest.run(model, data, issue_times)
    if "q0.5" not in fc:
        fc = conformal(fc, bt.quantiles, bt.warmup_days)

    job.out.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(job.out)
    if hasattr(model, "importance"):
        model.importance.rename("gain").to_csv(job.out.with_suffix(".importance.csv"))
    crossed = f", {model.crossed_hours:,} hours with crossing quantiles" if getattr(model, "quantiles", None) else ""
    return f"{job.country} {job.target} {job.model}: {len(fc):,} forecast hours{crossed}"


def _run_all(cfg: Config, jobs: list[Job]) -> None:
    with ProcessPoolExecutor() as pool:
        for msg in pool.map(_run, jobs, [cfg] * len(jobs)):
            print(msg)


def run_backtests(cfg: Config, countries: list[str], targets: list[str], models: list[str]) -> None:
    bt = cfg.backtest
    first_day = bt.test_start - timedelta(days=bt.warmup_days)
    jobs = [
        Job(c, t, m, first_day, bt.test_end, cfg.backtests_dir / c / t / f"{m}.parquet")
        for c, t, m in product(countries, targets, models)
    ]
    _run_all(cfg, jobs)


def tune(cfg: Config, countries: list[str], targets: list[str]) -> pd.DataFrame:
    """Try every grid combination for `lgbm` on the validation period (before the test year)."""
    tuning = cfg.lgbm["tuning"]
    start, end = date.fromisoformat(tuning["start"]), date.fromisoformat(tuning["end"])
    grid = [dict(zip(tuning["grid"], values)) for values in product(*tuning["grid"].values())]

    jobs = [
        Job(c, t, "lgbm", start, end, cfg.tuning_dir / c / t / f"{i}.parquet", {**combo, "retrain_days": tuning["retrain_days"]})
        for c, t in product(countries, targets)
        for i, combo in enumerate(grid)
    ]
    _run_all(cfg, jobs)

    rows = []
    for job in jobs:
        df = pd.read_parquet(job.out).dropna(subset=["actual", "forecast"])
        mae = (df["actual"] - df["forecast"]).abs().mean()
        rows.append({"country": job.country, "target": job.target, **job.overrides, "mae": mae})
    results = pd.DataFrame(rows).drop(columns="retrain_days")
    # Compare across countries on one scale: each run's MAE relative to the best run for that series.
    results["relative_mae"] = results["mae"] / results.groupby(["country", "target"])["mae"].transform("min")

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    results.round(3).to_csv(cfg.results_dir / "tuning.csv", index=False)
    params = list(tuning["grid"])
    return results.groupby(["target", *params])["relative_mae"].mean().round(3).reset_index().sort_values(["target", "relative_mae"])


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
    rows, by_hour, by_month, importance = [], [], [], []
    for country, target in product(countries, targets):
        frames = _load_test_period(cfg, country, target)
        if not frames:
            continue
        for name, df in frames.items():
            key = {"country": country, "target": target, "model": name}
            abs_err = (df["actual"] - df["forecast"]).abs()
            holiday = calendar(df.index, cfg.countries[country])["holiday"].astype(bool)
            s = score(df, cfg.backtest.quantiles, with_mape=target == "load_mw")
            rows.append({**key, **s, "mae_holiday": abs_err[holiday].mean(), "mae_normal": abs_err[~holiday].mean()})
            by_hour.append(abs_err.groupby(df.index.hour).mean().rename_axis("hour").reset_index(name="mae").assign(**key))
            by_month.append(abs_err.groupby(df.index.strftime("%Y-%m")).mean().rename_axis("month").reset_index(name="mae").assign(**key))

        for path in (cfg.backtests_dir / country / target).glob("*.importance.csv"):
            gain = pd.read_csv(path, index_col=0)["gain"]
            share = (gain / gain.sum() * 100).rename("gain_%").rename_axis("feature").reset_index()
            importance.append(share.assign(country=country, target=target, model=path.name.split(".")[0]))

    metrics = pd.DataFrame(rows)
    ref = metrics[metrics.model == REFERENCE].set_index(["country", "target"])["mae"]
    metrics["skill_%"] = (1 - metrics["mae"] / ref.reindex(pd.MultiIndex.from_frame(metrics[["country", "target"]])).to_numpy()) * 100

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    metrics.round(2).to_csv(cfg.results_dir / "metrics.csv", index=False)
    pd.concat(by_hour).round(1).to_csv(cfg.results_dir / "mae_by_hour.csv", index=False)
    pd.concat(by_month).round(1).to_csv(cfg.results_dir / "mae_by_month.csv", index=False)
    if importance:
        pd.concat(importance).round(2).to_csv(cfg.results_dir / "feature_importance.csv", index=False)
    return metrics


def _test_hours(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    bt = cfg.backtest
    df = df.loc[pd.Timestamp(bt.test_start, tz="UTC") : pd.Timestamp(bt.test_end, tz="UTC") + pd.Timedelta(hours=23)]
    return df.dropna(subset=["actual"])


def _interval_scores(df: pd.DataFrame, quantiles: list[float]) -> dict:
    y = df["actual"]
    inside = lambda lo, hi: float(((y >= df[lo]) & (y <= df[hi])).mean() * 100)
    return {
        "coverage_50_%": inside("q0.25", "q0.75"),
        "coverage_90_%": inside("q0.05", "q0.95"),
        "width_90": float((df["q0.95"] - df["q0.05"]).mean()),
        "winkler_90": winkler(df["q0.05"], df["q0.95"], y, alpha=0.1),
        "pinball": float(np.mean([pinball(y, df[f"q{q:g}"], q) for q in quantiles])),
    }


def intervals(cfg: Config, countries: list[str], targets: list[str]) -> pd.DataFrame:
    """Compare interval methods around the honest LightGBM, on the test year."""
    qs, window = cfg.backtest.quantiles, cfg.backtest.warmup_days
    rows, by_month = [], []
    for country, target in product(countries, targets):
        folder = cfg.backtests_dir / country / target
        point = pd.read_parquet(folder / "lgbm.parquet")
        qr = pd.read_parquet(folder / "lgbm_qr.parquet")

        methods = {("empirical", 0.0): conformal(point, qs, window), ("qr", 0.0): qr}
        for gamma in GAMMA_SENSITIVITY:
            methods["aci", gamma] = conformal(point, qs, window, gamma)
            methods["cqr", gamma] = conformal(qr, qs, window, gamma, base="quantiles")

        for (method, gamma), fc in methods.items():
            test = _test_hours(cfg, fc).dropna(subset=["q0.05", "q0.95"])
            key = {"country": country, "target": target, "method": method, "gamma": gamma}
            rows.append({**key, **_interval_scores(test, qs), "clipped_%": fc.attrs.get("clipped_share", 0) * 100})
            if gamma in (0.0, ACI_GAMMA):
                inside = (test["actual"] >= test["q0.05"]) & (test["actual"] <= test["q0.95"])
                monthly = inside.groupby(test.index.strftime("%Y-%m")).mean().mul(100)
                by_month.append(monthly.rename_axis("month").reset_index(name="coverage_90_%").assign(**key))

    table = pd.DataFrame(rows)
    table["main"] = table["gamma"].isin([0.0, ACI_GAMMA])
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(cfg.results_dir / "intervals.csv", index=False)
    pd.concat(by_month).round(1).to_csv(cfg.results_dir / "coverage_by_month.csv", index=False)
    return table[table["main"]].drop(columns="main")


def significance(cfg: Config, countries: list[str], targets: list[str]) -> pd.DataFrame:
    """Diebold-Mariano tests of lgbm against each other model, and bootstrap error bars on skill."""
    tests, cis = [], []
    for country, target in product(countries, targets):
        frames = _load_test_period(cfg, country, target)
        losses = {name: daily_mae(df) for name, df in frames.items()}
        for other in DM_COMPARISONS:
            stat, p = diebold_mariano(losses["lgbm"], losses[other])
            diff = (losses["lgbm"] - losses[other]).mean()
            tests.append({"country": country, "target": target, "model": "lgbm", "vs": other,
                          "mae_diff": diff, "dm_stat": stat, "p_value": p})
        for name, loss in losses.items():
            skill = 1 - loss.sum() / losses[REFERENCE].sum()
            ci = {f"ci{b}_{side}": v * 100 for b in (7, 14) for side, v in zip(("lo", "hi"), block_bootstrap_skill(loss, losses[REFERENCE], block=b))}
            cis.append({"country": country, "target": target, "model": name, "skill_%": skill * 100, **ci})

    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests["p_value"])
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    tests.to_csv(cfg.results_dir / "significance.csv", index=False, float_format="%.4g")
    pd.DataFrame(cis).round(1).to_csv(cfg.results_dir / "skill_ci.csv", index=False)
    return tests
