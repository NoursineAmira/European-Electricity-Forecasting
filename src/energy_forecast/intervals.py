"""Turning point forecasts into calibrated ranges.

Every method here is the same recipe with different settings:

    bound_q = base_q + (level-quantile of recent scores),   score = actual - base_q

per hour of day, over the last `window_days` days. The level starts at q. With gamma > 0 it
adapts (adaptive conformal inference): up after the truth lands above the bound, down otherwise.

- empirical: base = point forecast, gamma = 0
- aci:       base = point forecast, gamma > 0
- cqr:       base = the model's own q column, gamma > 0

Day T is forecast on day T-1 at the issue hour, when outcomes are known up to day T-2.
So both the score window and the level updates stop at T-2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _grid(s: pd.Series) -> pd.DataFrame:
    """Day × hour-of-day table."""
    return s.groupby([s.index.normalize(), s.index.hour]).first().unstack()


def conformal(
    fc: pd.DataFrame,
    quantiles: list[float],
    window_days: int,
    gamma: float = 0.0,
    base: str = "forecast",
) -> pd.DataFrame:
    actual = _grid(fc["actual"]).to_numpy()
    n_days, n_hours = actual.shape
    out = fc.copy()
    clipped = 0

    for q in quantiles:
        base_grid = _grid(fc["forecast"] if base == "forecast" else fc[f"q{q:g}"])
        b = base_grid.to_numpy()
        scores = actual - b
        bounds = np.full_like(b, np.nan)
        level = np.full(n_hours, q)

        for t in range(2, n_days):
            known = t - 2
            below = actual[known] <= bounds[known]
            observed = ~np.isnan(actual[known]) & ~np.isnan(bounds[known])
            level = np.where(observed, level + gamma * (q - below), level)

            past = scores[max(0, t - 1 - window_days) : t - 1]
            enough = np.sum(~np.isnan(past), axis=0) >= window_days // 2
            clipped += int(np.sum(enough & ((level < 0) | (level > 1))))
            for h in np.flatnonzero(enough):
                bounds[t, h] = b[t, h] + np.nanquantile(past[:, h], np.clip(level[h], 0, 1))

        stacked = pd.DataFrame(bounds, index=base_grid.index, columns=base_grid.columns).stack(future_stack=True)
        out[f"q{q:g}"] = stacked.reindex(pd.MultiIndex.from_arrays([fc.index.normalize(), fc.index.hour])).to_numpy()

    out.attrs["clipped_share"] = clipped / max(1, n_days * n_hours * len(quantiles))
    return out


def winkler(lo: pd.Series, hi: pd.Series, y: pd.Series, alpha: float) -> float:
    """Width of the (1 - alpha) range, plus 2/alpha times how far the truth fell outside it."""
    penalty = (2 / alpha) * ((lo - y).clip(lower=0) + (y - hi).clip(lower=0))
    return float(((hi - lo) + penalty).mean())
