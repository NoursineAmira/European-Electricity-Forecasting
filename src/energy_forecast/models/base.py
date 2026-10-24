from typing import ClassVar, Protocol

import pandas as pd


class Forecaster(Protocol):
    name: ClassVar[str]

    def predict(self, history: pd.Series, horizon: pd.DatetimeIndex) -> pd.Series:
        """Point forecast for `horizon`, using only `history` (which ends before the origin)."""
        ...
