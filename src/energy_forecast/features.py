"""Model inputs for each target hour, using only what is known at the issue time the day before.

The same function builds training rows and forecasting rows, so the two can't drift apart.
"""

from __future__ import annotations

import holidays
import pandas as pd

from .config import Country

HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)
LAGS_H = (48, 168, 336)


def issue_times(times: pd.DatetimeIndex, issue_hour: int) -> pd.DatetimeIndex:
    """When the forecast for each target hour was made: `issue_hour` UTC the day before."""
    return (times - DAY).normalize() + pd.Timedelta(hours=issue_hour)


def calendar(times: pd.DatetimeIndex, country: Country) -> pd.DataFrame:
    local = times.tz_convert(country.tz)
    day = local.tz_localize(None).normalize()
    public = pd.DatetimeIndex(
        list(holidays.country_holidays(country.code.upper(), years=range(day.year.min() - 1, day.year.max() + 2)))
    )

    def day_off(d: pd.DatetimeIndex):
        return (d.dayofweek >= 5) | d.isin(public)

    return pd.DataFrame(
        {
            "hour": local.hour,
            "dow": local.dayofweek,
            "doy": local.dayofyear,
            "month": local.month,
            "holiday": day.isin(public).astype(int),
            "bridge": (~day_off(day) & day_off(day - DAY) & day_off(day + DAY)).astype(int),
            "christmas": (((local.month == 12) & (local.day >= 24)) | ((local.month == 1) & (local.day == 1))).astype(int),
        },
        index=times,
    )


def build_features(
    y: pd.Series,
    weather: pd.DataFrame | None,
    times: pd.DatetimeIndex,
    country: Country,
    issue_hour: int,
) -> pd.DataFrame:
    """One row per target hour in `times`.

    `y` is the target on an hourly grid; values after each row's issue time are never read.
    `weather` (or None) must cover the 24 h before each target hour.
    """
    last_known = issue_times(times, issue_hour) - HOUR
    f = calendar(times, country)
    for h in LAGS_H:
        f[f"lag_{h}h"] = y.reindex(times - h * HOUR).to_numpy()
    f["last_value"] = y.reindex(last_known).to_numpy()
    f["mean_24h"] = y.rolling(24, min_periods=12).mean().reindex(last_known).to_numpy()

    if weather is not None:
        for col in weather:
            f[col] = weather[col].reindex(times).to_numpy()
        temp_24h = weather["temperature_2m"].rolling(24, min_periods=12).mean()
        f["temperature_24h"] = temp_24h.reindex(times).to_numpy()
    return f
