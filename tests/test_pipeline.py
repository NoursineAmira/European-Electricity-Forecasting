from datetime import date

import numpy as np
import pandas as pd

from energy_forecast.pipeline import clean, quarters, to_hourly


def test_quarters_cover_range_without_overlap():
    chunks = quarters(date(2024, 2, 15), date(2025, 1, 10))
    assert chunks[0] == (date(2024, 2, 15), date(2024, 3, 31))
    assert chunks[-1] == (date(2025, 1, 1), date(2025, 1, 10))
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert (next_start - prev_end).days == 1


def test_to_hourly_averages_quarter_hours():
    idx = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    df = pd.DataFrame({"load_mw": [1, 2, 3, 4, 10, 10, 10, 10]}, index=idx, dtype=float)
    out = to_hourly(df, date(2025, 1, 1), date(2025, 1, 1))
    assert len(out) == 24
    assert out["load_mw"].iloc[:2].tolist() == [2.5, 10.0]


def test_to_hourly_keeps_gaps_and_drops_out_of_range():
    idx = pd.DatetimeIndex(["2024-12-31 23:00", "2025-01-01 00:00", "2025-01-01 05:00"], tz="UTC")
    df = pd.DataFrame({"x": [9.0, 1.0, 2.0]}, index=idx)
    out = to_hourly(df, date(2025, 1, 1), date(2025, 1, 1))
    assert out.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert np.isnan(out.loc["2025-01-01 01:00", "x"].item())
    assert out["x"].notna().sum() == 2


def test_zero_load_is_dropped_before_averaging():
    idx = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    df = pd.DataFrame({"load_mw": [100, 0, 100, 100], "solar_mw": [0.0] * 4}, index=idx, dtype=float)
    out = to_hourly(clean("power", df), date(2025, 1, 1), date(2025, 1, 1))
    assert out["load_mw"].iloc[0] == 100
    assert out["solar_mw"].iloc[0] == 0


def test_to_hourly_spans_dst_change_in_utc():
    idx = pd.date_range("2025-03-30", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"x": range(24)}, index=idx, dtype=float)
    out = to_hourly(df, date(2025, 3, 30), date(2025, 3, 30))
    assert len(out) == 24 and out["x"].notna().all()
