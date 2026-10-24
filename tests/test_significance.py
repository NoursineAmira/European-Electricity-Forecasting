import numpy as np
import pandas as pd

from energy_forecast.significance import block_bootstrap_skill, diebold_mariano, holm

DAYS = pd.date_range("2025-01-01", periods=365, freq="D", tz="UTC")


def test_dm_detects_a_real_gap_and_not_noise():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.gamma(2, 50, len(DAYS)), index=DAYS)
    better = base - 10 + rng.normal(0, 5, len(DAYS))
    same = base + rng.normal(0, 5, len(DAYS))
    stat, p = diebold_mariano(better, base)
    assert stat < 0 and p < 1e-6
    assert diebold_mariano(same, base)[1] > 0.05


def test_holm():
    p = pd.Series([0.01, 0.04, 0.03], index=["a", "b", "c"])
    assert holm(p).round(6).tolist() == [0.03, 0.06, 0.06]


def test_bootstrap_interval_contains_skill():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.gamma(2, 50, len(DAYS)), index=DAYS)
    assert np.allclose(block_bootstrap_skill(ref * 0.5, ref), 0.5)
    noisy = ref * 0.5 * rng.uniform(0.5, 1.5, len(DAYS))
    lo, hi = block_bootstrap_skill(noisy, ref)
    skill = 1 - noisy.sum() / ref.sum()
    assert lo < skill < hi and hi - lo < 0.2
