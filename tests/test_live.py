import pandas as pd

from energy_forecast.live import fill_actuals, merge_history


def _issue(issued: str, value: float) -> pd.DataFrame:
    hours = pd.date_range(pd.Timestamp(issued, tz="UTC") + pd.Timedelta(days=1), periods=2, freq="h")
    return pd.DataFrame({"timestamp": hours, "forecast": value, "country": "de", "target": "load_mw", "issued": issued})


def test_a_later_run_replaces_the_same_day_and_keeps_earlier_days():
    history = pd.concat([_issue("2026-10-20", 1.0), _issue("2026-10-21", 2.0)], ignore_index=True)
    merged = merge_history(history, _issue("2026-10-21", 3.0))
    assert merged.groupby("issued")["forecast"].first().to_dict() == {"2026-10-20": 1.0, "2026-10-21": 3.0}
    assert len(merged) == 4


def test_first_run_starts_the_history():
    assert merge_history(pd.DataFrame(), _issue("2026-10-20", 1.0)).equals(_issue("2026-10-20", 1.0))


def test_actuals_fill_in_where_known():
    history = _issue("2026-10-20", 1.0)
    truth = pd.Series([5.0], index=history["timestamp"][:1])
    out = fill_actuals(history, {("de", "load_mw"): truth})
    assert out["actual"].iloc[0] == 5.0 and pd.isna(out["actual"].iloc[1])
