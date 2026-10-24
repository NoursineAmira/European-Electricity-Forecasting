"""Is one forecaster really better than another, or was the test year lucky?

Both tools work on *daily* losses: the 24 hours of one forecast share the same mistakes,
so treating them as 24 independent samples would make every difference look significant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def daily_mae(df: pd.DataFrame) -> pd.Series:
    err = (df["actual"] - df["forecast"]).abs()
    return err.groupby(err.index.normalize()).mean()


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, lag: int = 7) -> tuple[float, float]:
    """Test whether mean(loss_a - loss_b) differs from zero.

    Newey-West variance (Bartlett weights) allows neighbouring days to be alike; the
    Harvey-Leybourne-Newbold factor and a t distribution fix small-sample optimism.
    Returns (statistic, two-sided p-value). Negative statistic: A has the lower loss.
    """
    d = (loss_a - loss_b).dropna().to_numpy()
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(var / n) * np.sqrt((n - 1) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def holm(pvalues: pd.Series) -> pd.Series:
    """Holm correction: keeps the chance of *any* false 'significant' result at the chosen level."""
    order = pvalues.sort_values()
    m = len(order)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * order.to_numpy()).clip(max=1)
    return pd.Series(adjusted, index=order.index).reindex(pvalues.index)


def block_bootstrap_skill(
    loss: pd.Series, ref_loss: pd.Series, block: int = 7, reps: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% interval for skill = 1 - sum(loss) / sum(ref_loss), resampling whole blocks of days."""
    pair = pd.concat([loss, ref_loss], axis=1).dropna().to_numpy()
    n = len(pair)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(reps, -(-n // block)))
    idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]
    sums = pair[idx].sum(axis=1)
    skills = 1 - sums[:, 0] / sums[:, 1]
    lo, hi = np.percentile(skills, [2.5, 97.5])
    return float(lo), float(hi)
