from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy_forecast import backtest
from energy_forecast.config import load_config
from energy_forecast.metrics import pinball, score
from energy_forecast.models import MODEL_NAMES, make

POISON = 1e12
CFG = load_config()
ORIGIN = pd.Timestamp("2021-02-20 10:00", tz="UTC")


def synthetic_data(days: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=24 * days, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    temp = 5 + 5 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 1, len(idx))
    data = pd.DataFrame(
        {
            "load_mw": 100 + 20 * np.sin(2 * np.pi * idx.hour / 24) - 2 * temp + rng.normal(0, 1, len(idx)),
            "price_eur_mwh": 80 + rng.normal(0, 10, len(idx)),
        },
        index=idx,
    )
    for v in CFG.weather_variables:
        data[v] = temp if v == "temperature_2m" else rng.uniform(0, 100, len(idx))
        data[f"{v}_fc"] = data[v] + rng.normal(0, 0.5, len(idx))
    return data


def _model(name: str):
    return make(name, "load_mw", CFG.countries["de"], CFG, {"n_estimators": 20, "n_jobs": 1})


def test_origins_and_targets():
    issue = backtest.origins(date(2025, 3, 1), date(2025, 3, 3), issue_hour=10)
    assert list(issue.strftime("%m-%d %H")) == ["02-28 10", "03-01 10", "03-02 10"]
    hours = backtest.target_hours(issue[0])
    assert hours[0] == pd.Timestamp("2025-03-01", tz="UTC") and len(hours) == 24


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_never_see_the_future(name):
    """Poison everything after the origin except the columns a model may see: no forecast may change."""
    clean = synthetic_data()
    model = _model(name)
    poisoned = clean.copy()
    hidden = [c for c in clean if c not in model.future_columns]
    poisoned.loc[ORIGIN:, hidden] = POISON

    issue = pd.DatetimeIndex([ORIGIN])
    expected = backtest.run(_model(name), clean, issue)["forecast"]
    got = backtest.run(model, poisoned, issue)["forecast"]
    assert got.notna().all()
    pd.testing.assert_series_equal(got, expected)


def test_history_and_future_windows():
    seen = {}

    class Spy:
        name, target, future_columns = "spy", "load_mw", ["temperature_2m_fc"]

        def predict(self, history, future, horizon):
            seen.update(last=history.index[-1], future=(future.index[0], future.index[-1]), cols=list(future))
            return pd.Series(0.0, index=horizon)

    backtest.run(Spy(), synthetic_data(), pd.DatetimeIndex([ORIGIN]))
    assert seen["last"] == ORIGIN - pd.Timedelta(hours=1)
    assert seen["future"] == (ORIGIN, pd.Timestamp("2021-02-21 23:00", tz="UTC"))
    assert seen["cols"] == ["temperature_2m_fc"]


def test_pinball_and_score():
    y = pd.Series([10.0, 10.0])
    assert pinball(y, pd.Series([8.0, 12.0]), 0.5) == 1.0
    df = pd.DataFrame({"actual": [10.0, 20.0], "forecast": [12.0, 20.0], "q0.05": [0.0, 0.0], "q0.95": [11.0, 30.0]})
    s = score(df, [0.05, 0.95], with_mape=True)
    assert s["mae"] == 1.0 and s["coverage_90_%"] == 100.0 and s["mape_%"] == 10.0
