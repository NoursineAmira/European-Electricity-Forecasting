import numpy as np
import pandas as pd

from energy_forecast.intervals import conformal, winkler


def _frame(days: int, errors: np.ndarray | None = None) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=24 * days, freq="h", tz="UTC")
    err = np.ones(len(idx)) if errors is None else errors
    return pd.DataFrame({"forecast": 0.0, "actual": err}, index=idx)


def test_empirical_uses_only_known_errors():
    fc = _frame(10)
    fc.loc["2025-01-09":, "actual"] = 100.0  # errors on day 9 onward must not reach day 10
    out = conformal(fc, [0.5], window_days=4)
    assert (out.loc["2025-01-10", "q0.5"] == 1.0).all()
    assert out.loc["2025-01-01":"2025-01-03", "q0.5"].isna().all()


def test_aci_reacts_to_a_miss_two_days_later_not_before():
    rng = np.random.default_rng(1)
    fc = _frame(30, rng.normal(0, 1, 24 * 30))
    shocked = fc.copy()
    shocked.loc["2025-01-15", "actual"] = 50.0

    a = conformal(fc, [0.95], window_days=10, gamma=0.1)["q0.95"]
    b = conformal(shocked, [0.95], window_days=10, gamma=0.1)["q0.95"]
    assert a.loc[:"2025-01-16"].equals(b.loc[:"2025-01-16"])
    assert (b.loc["2025-01-17"] > a.loc["2025-01-17"]).all()


def test_aci_fixes_coverage_when_errors_drift():
    days = 400
    rng = np.random.default_rng(0)
    spread = np.repeat(np.linspace(1, 4, days), 24)  # errors keep getting bigger
    fc = _frame(days, rng.normal(0, 1, 24 * days) * spread)

    def coverage(out):
        tail = out.iloc[24 * 100 :]
        return ((tail["actual"] >= tail["q0.05"]) & (tail["actual"] <= tail["q0.95"])).mean()

    empirical = coverage(conformal(fc, [0.05, 0.95], window_days=56))
    aci = coverage(conformal(fc, [0.05, 0.95], window_days=56, gamma=0.005))
    assert empirical < 0.88
    assert abs(aci - 0.90) < abs(empirical - 0.90)


def test_winkler_penalises_misses():
    y = pd.Series([5.0, 12.0])
    lo, hi = pd.Series([0.0, 0.0]), pd.Series([10.0, 10.0])
    # widths 10 and 10; second point is 2 above, penalty 2 / 0.1 * 2 = 40
    assert winkler(lo, hi, y, alpha=0.1) == (10 + 50) / 2
