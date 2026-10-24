from datetime import date

import pandas as pd

from .http import get_json

BASE = "https://api.energy-charts.info"

POWER_COLUMNS = {
    "Load": "load_mw",
    "Solar": "solar_mw",
    "Wind onshore": "wind_onshore_mw",
    "Wind offshore": "wind_offshore_mw",
}


def _index(unix_seconds: list[int]) -> pd.DatetimeIndex:
    return pd.to_datetime(unix_seconds, unit="s", utc=True).rename("timestamp")


def fetch_power(country: str, start: date, end: date) -> pd.DataFrame:
    data = get_json(f"{BASE}/public_power", {"country": country, "start": start, "end": end})
    series = {
        POWER_COLUMNS[p["name"]]: p["data"]
        for p in data["production_types"]
        if p["name"] in POWER_COLUMNS
    }
    return pd.DataFrame(series, index=_index(data["unix_seconds"]), dtype="float64")


def fetch_price(bzn: str, start: date, end: date) -> pd.DataFrame:
    data = get_json(f"{BASE}/price", {"bzn": bzn, "start": start, "end": end})
    return pd.DataFrame(
        {"price_eur_mwh": data["price"]}, index=_index(data["unix_seconds"]), dtype="float64"
    )
