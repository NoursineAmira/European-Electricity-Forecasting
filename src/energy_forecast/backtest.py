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
        rows.append(pd.DataFrame({"origin": origin, "forecast": forecast.to_numpy()}, index=horizon))
    out = pd.concat(rows).rename_axis("timestamp")
    return out.assign(actual=data[model.target].reindex(out.index).to_numpy())


def add_quantiles(fc: pd.DataFrame, quantiles: list[float], window_days: int) -> pd.DataFrame:
    """Intervals from the model's own past errors, per hour of day.

    For target day T, the forecast is issued on T-1, when actuals are known up to T-2.
    So only errors from target days T-1-window .. T-2 are used: hence the shift(2).
    """
    errors = (fc["actual"] - fc["forecast"]).rename("error").to_frame()
    errors["day"] = errors.index.normalize()
    errors["hour"] = errors.index.hour
    grid = errors.pivot(index="day", columns="hour", values="error")

    out = fc.copy()
    day, hour = fc.index.normalize(), fc.index.hour
    rolling = grid.rolling(window_days, min_periods=window_days // 2)
    for q in quantiles:
        q_err = rolling.quantile(q).shift(2).stack().rename("e")
        lookup = q_err.reindex(pd.MultiIndex.from_arrays([day, hour])).to_numpy()
        out[f"q{q:g}"] = fc["forecast"].to_numpy() + lookup
    return out
