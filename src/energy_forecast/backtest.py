from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .models.base import Forecaster

HOUR = pd.Timedelta(hours=1)


def origins(first_day: date, last_target_day: date, issue_hour: int) -> pd.DatetimeIndex:
    """Issue times: `issue_hour` UTC on each day whose next day is a target day."""
    days = pd.date_range(first_day - timedelta(days=1), last_target_day - timedelta(days=1), tz="UTC")
    return days + pd.Timedelta(hours=issue_hour)


def target_hours(origin: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(origin.normalize() + pd.Timedelta(days=1), periods=24, freq="h")


def run(model: Forecaster, data: pd.DataFrame, issue_times: pd.DatetimeIndex) -> pd.DataFrame:
    """Replay the past: for each origin, the model sees all data before it,
    and after it only the columns it declares in `future_columns` (e.g. weather forecasts)."""
    rows = []
    for origin in issue_times:
        horizon = target_hours(origin)
        history = data.loc[: origin - HOUR]
        future = data.loc[origin : horizon[-1], model.future_columns]
        forecast = model.predict(history, future, horizon)
        frame = forecast.to_frame("forecast") if isinstance(forecast, pd.Series) else forecast
        rows.append(frame.assign(origin=origin))
    out = pd.concat(rows).rename_axis("timestamp")
    return out.assign(actual=data[model.target].reindex(out.index).to_numpy())
