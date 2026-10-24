from datetime import date

import pandas as pd

from .http import get_json

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def fetch_weather(
    cities: dict[str, tuple[float, float]],
    start: date,
    end: date,
    variables: list[str],
    forecast: bool = False,
) -> pd.DataFrame:
    """Hourly weather averaged over the given cities (one request for all of them).

    With `forecast=True`, returns the forecast issued two days before each hour
    (columns suffixed `_fc`) instead of what was observed.
    """
    lats, lons = zip(*cities.values())
    api_vars = [f"{v}_previous_day2" for v in variables] if forecast else variables
    data = get_json(
        PREVIOUS_RUNS_URL if forecast else ARCHIVE_URL,
        {
            "latitude": ",".join(map(str, lats)),
            "longitude": ",".join(map(str, lons)),
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(api_vars),
            "timezone": "UTC",
        },
    )
    locations = data if isinstance(data, list) else [data]

    frames = []
    for loc in locations:
        hourly = loc["hourly"]
        index = pd.to_datetime(hourly["time"], utc=True).rename("timestamp")
        columns = {v: hourly[a] for v, a in zip(variables, api_vars)}
        frames.append(pd.DataFrame(columns, index=index, dtype="float64"))
    weather = pd.concat(frames).groupby(level="timestamp").mean()
    return weather.add_suffix("_fc") if forecast else weather
