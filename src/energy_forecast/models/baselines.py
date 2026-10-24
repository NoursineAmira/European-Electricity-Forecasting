import numpy as np
import pandas as pd
from statsforecast.models import MSTL, AutoETS

DAY = pd.Timedelta(days=1)
WEEK = pd.Timedelta(weeks=1)


def _lagged(history: pd.Series, horizon: pd.DatetimeIndex, lag: pd.Timedelta) -> pd.Series:
    return pd.Series(history.reindex(horizon - lag).to_numpy(), index=horizon)


class Baseline:
    """Univariate models: they only look at the target's own past."""

    name = ""
    future_columns: list[str] = []

    def __init__(self, target: str):
        self.target = target

    def predict(self, history: pd.DataFrame, future: pd.DataFrame, horizon: pd.DatetimeIndex) -> pd.Series:
        return self._predict(history[self.target], horizon)


class NaiveDaily(Baseline):
    name = "naive_daily"

    def _predict(self, y, horizon):
        return _lagged(y, horizon, 2 * DAY)


class NaiveWeekly(Baseline):
    name = "naive_weekly"

    def _predict(self, y, horizon):
        return _lagged(y, horizon, WEEK)


class WeeklyMean(Baseline):
    name = "weekly_mean"

    def _predict(self, y, horizon):
        lags = [_lagged(y, horizon, k * WEEK) for k in range(1, 5)]
        return pd.concat(lags, axis=1).mean(axis=1)


class MSTLModel(Baseline):
    name = "mstl"
    window = 8 * WEEK

    def _predict(self, y, horizon):
        y = y.loc[y.index[-1] - self.window :].interpolate().bfill()
        steps = int((horizon[-1] - y.index[-1]) / pd.Timedelta(hours=1))
        model = MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"))
        mean = model.fit(y.to_numpy(np.float64)).predict(h=steps)["mean"]
        return pd.Series(mean[-len(horizon) :], index=horizon)
