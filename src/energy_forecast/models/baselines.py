import numpy as np
import pandas as pd
from statsforecast.models import MSTL, AutoETS

DAY = pd.Timedelta(days=1)
WEEK = pd.Timedelta(weeks=1)


def _lagged(history: pd.Series, horizon: pd.DatetimeIndex, lag: pd.Timedelta) -> pd.Series:
    return pd.Series(history.reindex(horizon - lag).to_numpy(), index=horizon)


class NaiveDaily:
    name = "naive_daily"

    def predict(self, history, horizon):
        return _lagged(history, horizon, 2 * DAY)


class NaiveWeekly:
    name = "naive_weekly"

    def predict(self, history, horizon):
        return _lagged(history, horizon, WEEK)


class WeeklyMean:
    name = "weekly_mean"

    def predict(self, history, horizon):
        lags = [_lagged(history, horizon, k * WEEK) for k in range(1, 5)]
        return pd.concat(lags, axis=1).mean(axis=1)


class MSTLModel:
    name = "mstl"
    window = 8 * WEEK

    def predict(self, history, horizon):
        y = history.loc[history.index[-1] - self.window :].interpolate().bfill()
        steps = int((horizon[-1] - y.index[-1]) / pd.Timedelta(hours=1))
        model = MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"))
        mean = model.fit(y.to_numpy(np.float64)).predict(h=steps)["mean"]
        return pd.Series(mean[-len(horizon) :], index=horizon)
