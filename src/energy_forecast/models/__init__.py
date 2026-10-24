from ..config import Config, Country
from .base import Forecaster
from .baselines import MSTLModel, NaiveDaily, NaiveWeekly, WeeklyMean
from .lgbm import LGBM, WEATHER_MODES

BASELINES = {m.name: m for m in (NaiveDaily, NaiveWeekly, WeeklyMean, MSTLModel)}
MODEL_NAMES = [*BASELINES, *WEATHER_MODES]


def make(name: str, target: str, country: Country, cfg: Config, overrides: dict | None = None) -> Forecaster:
    if name in WEATHER_MODES:
        return LGBM(name, target, country, cfg, overrides)
    return BASELINES[name](target)
