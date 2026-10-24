from typing import Protocol

import pandas as pd


class Forecaster(Protocol):
    name: str
    target: str
    future_columns: list[str]  # columns the backtest may show for the hours after the origin

    def predict(self, history: pd.DataFrame, future: pd.DataFrame, horizon: pd.DatetimeIndex) -> pd.Series:
        """Point forecast for `horizon`.

        `history`: every column, up to the hour before the origin.
        `future`: only `future_columns`, from the origin to the end of `horizon`.
        """
        ...
