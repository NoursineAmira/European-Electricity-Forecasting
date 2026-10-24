from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import Config, Country
from ..features import build_features

HOUR = pd.Timedelta(hours=1)

# How each version gets tomorrow's weather.
WEATHER_MODES = {"lgbm_noweather": "none", "lgbm_oracle": "observed", "lgbm": "forecast", "lgbm_qr": "forecast"}
QUANTILE_MODELS = {"lgbm_qr"}


class LGBM:
    def __init__(self, name: str, target: str, country: Country, cfg: Config, overrides: dict | None = None):
        spec = cfg.lgbm["targets"][target]
        settings = {**cfg.lgbm["base_params"], **spec["params"], **(overrides or {})}

        self.name, self.target, self.country = name, target, country
        self.mode = WEATHER_MODES[name]
        self.quantiles = cfg.backtest.quantiles if name in QUANTILE_MODELS else None
        self.train_start = pd.Timestamp(settings.pop("train_start", spec["train_start"]), tz="UTC")
        self.retrain_every = pd.Timedelta(days=settings.pop("retrain_days", cfg.lgbm["retrain_days"]))
        self.rounds = settings.pop("n_estimators")
        self.params = settings
        self.issue_hour = cfg.backtest.issue_hour_utc
        self.weather_vars = list(cfg.weather_variables)
        self.future_columns = {
            "none": [],
            "observed": self.weather_vars,
            "forecast": [f"{v}_fc" for v in self.weather_vars],
        }[self.mode]
        self._models: dict = {}
        self._fitted_at = None
        self.crossed_hours = 0

    def _features(self, y, weather, times):
        return build_features(y, weather, times, self.country, self.issue_hour)

    def _fit(self, history: pd.DataFrame, origin: pd.Timestamp) -> None:
        y = history[self.target]
        weather = None if self.mode == "none" else history[self.weather_vars]
        train = y.loc[self.train_start :].dropna()
        X = self._features(y, weather, train.index)

        if self.quantiles:
            params = {q: {**self.params, "objective": "quantile", "alpha": q} for q in self.quantiles}
        else:
            params = {None: self.params}
        self._models = {k: lgb.train(p, lgb.Dataset(X, train), num_boost_round=self.rounds) for k, p in params.items()}
        self._fitted_at = origin
        main = self._models[0.5 if self.quantiles else None]
        self.importance = pd.Series(main.feature_importance("gain"), index=X.columns)

    def predict(self, history: pd.DataFrame, future: pd.DataFrame, horizon: pd.DatetimeIndex):
        origin = history.index[-1] + HOUR
        if not self._models or origin - self._fitted_at >= self.retrain_every:
            self._fit(history, origin)

        y = history[self.target].reindex(history.index.append(future.index))
        weather = None
        if self.mode != "none":
            ahead = future[self.future_columns].set_axis(self.weather_vars, axis=1)
            weather = pd.concat([history[self.weather_vars], ahead])
        X = self._features(y, weather, horizon)

        if not self.quantiles:
            return pd.Series(self._models[None].predict(X), index=horizon)

        raw = np.column_stack([self._models[q].predict(X) for q in self.quantiles])
        ordered = np.sort(raw, axis=1)  # separately trained quantiles can cross; sorting is the standard fix
        self.crossed_hours += int((raw != ordered).any(axis=1).sum())
        out = pd.DataFrame(ordered, index=horizon, columns=[f"q{q:g}" for q in self.quantiles])
        return out.assign(forecast=out["q0.5"])
