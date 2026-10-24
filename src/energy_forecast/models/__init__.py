from .baselines import MSTLModel, NaiveDaily, NaiveWeekly, WeeklyMean
from .base import Forecaster

MODELS: dict[str, type[Forecaster]] = {
    m.name: m for m in (NaiveDaily, NaiveWeekly, WeeklyMean, MSTLModel)
}
