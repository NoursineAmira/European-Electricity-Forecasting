import pandas as pd

from energy_forecast.features import build_features, calendar
from test_backtest import CFG, ORIGIN, synthetic_data

DE = CFG.countries["de"]


def _day(s: str) -> pd.DatetimeIndex:
    return pd.date_range(f"{s} 00:00", periods=24, freq="h", tz=DE.tz).tz_convert("UTC")


def test_holidays_bridges_christmas():
    assert calendar(_day("2025-04-21"), DE)["holiday"].all()  # Easter Monday
    assert calendar(_day("2025-05-30"), DE)["bridge"].all()  # Friday after Ascension
    assert not calendar(_day("2025-06-03"), DE)[["holiday", "bridge"]].any().any()
    assert calendar(_day("2025-12-24"), DE)["christmas"].all()


def test_hour_is_local_through_clock_changes():
    winter = calendar(pd.DatetimeIndex(["2025-01-15 12:00"], tz="UTC"), DE)["hour"].item()
    summer = calendar(pd.DatetimeIndex(["2025-07-15 12:00"], tz="UTC"), DE)["hour"].item()
    assert (winter, summer) == (13, 14)


def test_training_row_equals_forecasting_row():
    """A row built from the full past (training) must equal the row built at forecast time."""
    data = synthetic_data()
    weather_cols = CFG.weather_variables
    horizon = pd.date_range(ORIGIN.normalize() + pd.Timedelta(days=1), periods=24, freq="h")

    as_training = build_features(data["load_mw"], data[weather_cols], horizon, DE, 10)

    history = data.loc[: ORIGIN - pd.Timedelta(hours=1)]
    y = history["load_mw"].reindex(data.loc[: horizon[-1]].index)
    as_forecast = build_features(y, data.loc[: horizon[-1], weather_cols], horizon, DE, 10)

    pd.testing.assert_frame_equal(as_training, as_forecast)
    assert as_training.notna().all().all()
