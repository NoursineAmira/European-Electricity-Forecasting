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
class Config:
    countries: dict[str, Country]
    start: date
    end: date
    weather_variables: list[str]
    raw_dir: Path
    processed_dir: Path


def load_config(path: Path | None = None) -> Config:
    path = path or ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text())

    countries = {
        code: Country(code, c["name"], c["bzn"], {k: tuple(v) for k, v in c["cities"].items()})
        for code, c in cfg["countries"].items()
    }
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    end = date.fromisoformat(cfg["end"]) if cfg.get("end") else yesterday

    return Config(
        countries=countries,
        start=date.fromisoformat(cfg["start"]),
        end=end,
        weather_variables=cfg["weather_variables"],
        raw_dir=ROOT / cfg["paths"]["raw"],
        processed_dir=ROOT / cfg["paths"]["processed"],
    )
