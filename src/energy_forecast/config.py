from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Country:
    code: str
    name: str
    bzn: str
    cities: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class Backtest:
    test_start: date
    test_end: date
    warmup_days: int
    issue_hour_utc: int
    quantiles: list[float]


@dataclass(frozen=True)
class Config:
    countries: dict[str, Country]
    start: date
    end: date
    weather_variables: list[str]
    backtest: Backtest
    raw_dir: Path
    processed_dir: Path
    backtests_dir: Path
    results_dir: Path


def load_config(path: Path | None = None) -> Config:
    path = path or ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text())

    countries = {
        code: Country(code, c["name"], c["bzn"], {k: tuple(v) for k, v in c["cities"].items()})
        for code, c in cfg["countries"].items()
    }
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    end = date.fromisoformat(cfg["end"]) if cfg.get("end") else yesterday
    bt = cfg["backtest"]
    paths = cfg["paths"]

    return Config(
        countries=countries,
        start=date.fromisoformat(cfg["start"]),
        end=end,
        weather_variables=cfg["weather_variables"],
        backtest=Backtest(
            test_start=date.fromisoformat(bt["test_start"]),
            test_end=date.fromisoformat(bt["test_end"]),
            warmup_days=bt["warmup_days"],
            issue_hour_utc=bt["issue_hour_utc"],
            quantiles=bt["quantiles"],
        ),
        raw_dir=ROOT / paths["raw"],
        processed_dir=ROOT / paths["processed"],
        backtests_dir=ROOT / paths["backtests"],
        results_dir=ROOT / paths["results"],
    )
