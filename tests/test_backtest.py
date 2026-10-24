from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy_forecast import backtest
from energy_forecast.metrics import pinball, score
from energy_forecast.models import MODELS

POISON = 1e12


def _series(days: int = 70) -> pd.Series:
    idx = pd.date_range("2025-01-01", periods=24 * days, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    daily = 100 + 20 * np.sin(2 * np.pi * idx.hour / 24)
    return pd.Series(daily + rng.normal(0, 1, len(idx)), index=idx)


def test_origins_and_targets():
    issue = backtest.origins(date(2025, 3, 1), date(2025, 3, 3), issue_hour=10)
    assert list(issue.strftime("%m-%d %H")) == ["02-28 10", "03-01 10", "03-02 10"]
    hours = backtest.target_hours(issue[0])
    assert hours[0] == pd.Timestamp("2025-03-01", tz="UTC") and len(hours) == 24


@pytest.mark.parametrize("name", list(MODELS))
def test_models_never_see_the_future(name):
    s = _series()
    origin = pd.Timestamp("2025-03-05 10:00", tz="UTC")
    poisoned = s.copy()
    poisoned[origin:] = POISON
    fc = backtest.run(MODELS[name](), poisoned, pd.DatetimeIndex([origin]))
    assert fc["forecast"].notna().all()
    assert (fc["forecast"].abs() < POISON / 10).all()


def test_history_ends_just_before_origin():
    seen = []

    class Spy:
        name = "spy"

        def predict(self, history, horizon):
            seen.append(history.index[-1])
            return pd.Series(0.0, index=horizon)

    origin = pd.Timestamp("2025-03-05 10:00", tz="UTC")
    backtest.run(Spy(), _series(), pd.DatetimeIndex([origin]))
    assert seen == [pd.Timestamp("2025-03-05 09:00", tz="UTC")]


def test_quantiles_use_only_known_errors():
    idx = pd.date_range("2025-01-01", periods=24 * 10, freq="h", tz="UTC")
    fc = pd.DataFrame({"forecast": 0.0, "actual": 1.0}, index=idx)
    fc.loc["2025-01-09":, "actual"] = 100.0  # errors on day 9 onward must not reach day 10
    out = backtest.add_quantiles(fc, [0.5], window_days=4)
    assert (out.loc["2025-01-10", "q0.5"] == 1.0).all()
    assert out.loc["2025-01-01":"2025-01-03", "q0.5"].isna().all()


def test_pinball_and_score():
    y = pd.Series([10.0, 10.0])
    assert pinball(y, pd.Series([8.0, 12.0]), 0.5) == 1.0
    df = pd.DataFrame({"actual": [10.0, 20.0], "forecast": [12.0, 20.0], "q0.05": [0.0, 0.0], "q0.95": [11.0, 30.0]})
    s = score(df, [0.05, 0.95], with_mape=True)
    assert s["mae"] == 1.0 and s["coverage_90_%"] == 100.0 and s["mape_%"] == 10.0
