from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from ..config import Config, Country
from ..features import build_features

HOUR = pd.Timedelta(hours=1)

# How each version gets tomorrow's weather.
WEATHER_MODES = {"lgbm_noweather": "none", "lgbm_oracle": "observed", "lgbm": "forecast"}


class LGBM:
    def __init__(self, name: str, target: str, country: Country, cfg: Config, overrides: dict | None = None):
        spec = cfg.lgbm["targets"][target]
        settings = {**cfg.lgbm["base_params"], **spec["params"], **(overrides or {})}

        self.name, self.target, self.country = name, target, country
        self.mode = WEATHER_MODES[name]
        self.train_start = pd.Timestamp(settings.pop("train_start", spec["train_start"]), tz="UTC")
        self.retrain_every = pd.Timedelta(days=settings.pop("retrain_days", cfg.lgbm["retrain_days"]))
        self.params = settings
        self.issue_hour = cfg.backtest.issue_hour_utc
        self.weather_vars = list(cfg.weather_variables)
        self.future_columns = {
            "none": [],
            "observed": self.weather_vars,
            "forecast": [f"{v}_fc" for v in self.weather_vars],
        }[self.mode]
        self._model = None
        self._fitted_at = None

    def _features(self, y, weather, times):
        return build_features(y, weather, times, self.country, self.issue_hour)

    def _fit(self, history: pd.DataFrame, origin: pd.Timestamp) -> None:
        y = history[self.target]
        weather = None if self.mode == "none" else history[self.weather_vars]
        train = y.loc[self.train_start :].dropna()
        X = self._features(y, weather, train.index)
        params = dict(self.params)
        rounds = params.pop("n_estimators")
        self._model = lgb.train(params, lgb.Dataset(X, train), num_boost_round=rounds)
        self._fitted_at = origin
        self.importance = pd.Series(self._model.feature_importance("gain"), index=X.columns)

    def predict(self, history: pd.DataFrame, future: pd.DataFrame, horizon: pd.DatetimeIndex) -> pd.Series:
        origin = history.index[-1] + HOUR
        if self._model is None or origin - self._fitted_at >= self.retrain_every:
            self._fit(history, origin)

        y = history[self.target].reindex(history.index.append(future.index))
        weather = None
        if self.mode != "none":
            ahead = future[self.future_columns].set_axis(self.weather_vars, axis=1)
            weather = pd.concat([history[self.weather_vars], ahead])
        X = self._features(y, weather, horizon)
        return pd.Series(self._model.predict(X), index=horizon)
