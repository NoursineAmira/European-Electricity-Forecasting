from datetime import date

import pandas as pd

from .http import get_json

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(
    cities: dict[str, tuple[float, float]], start: date, end: date, variables: list[str]
) -> pd.DataFrame:
    """Hourly weather averaged over the given cities (one request for all of them)."""
    lats, lons = zip(*cities.values())
    data = get_json(
        ARCHIVE_URL,
        {
            "latitude": ",".join(map(str, lats)),
            "longitude": ",".join(map(str, lons)),
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(variables),
            "timezone": "UTC",
        },
    )
    locations = data if isinstance(data, list) else [data]

    frames = []
    for loc in locations:
        hourly = loc["hourly"]
        index = pd.to_datetime(hourly["time"], utc=True).rename("timestamp")
        frames.append(pd.DataFrame({v: hourly[v] for v in variables}, index=index, dtype="float64"))
    return pd.concat(frames).groupby(level="timestamp").mean()
