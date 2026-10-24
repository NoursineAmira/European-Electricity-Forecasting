import numpy as np
import pandas as pd


def pinball(y: pd.Series, pred: pd.Series, q: float) -> float:
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def score(df: pd.DataFrame, quantiles: list[float], with_mape: bool) -> pd.Series:
    """Scores for one model's backtest rows. Rows missing an actual or forecast are skipped."""
    df = df.dropna(subset=["actual", "forecast"])
    err = df["actual"] - df["forecast"]
    lo, hi = df[f"q{min(quantiles):g}"], df[f"q{max(quantiles):g}"]
    s = {
        "hours": len(df),
        "mae": err.abs().mean(),
        "rmse": np.sqrt((err**2).mean()),
        "mape_%": (err.abs() / df["actual"].abs()).mean() * 100 if with_mape else np.nan,
        "pinball": np.mean([pinball(df["actual"], df[f"q{q:g}"], q) for q in quantiles]),
        "coverage_90_%": ((df["actual"] >= lo) & (df["actual"] <= hi)).mean() * 100,
        "width_90": (hi - lo).mean(),
    }
    return pd.Series(s)
