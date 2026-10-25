import time

import requests

PAUSE_S = 0.5


def get_json(url: str, params: dict, retries: int = 4) -> dict | list:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            time.sleep(PAUSE_S)
            return resp.json()
        except requests.RequestException as err:
            if attempt == retries - 1:
                raise
            wait = 2**attempt * 5
            print(f"retry in {wait}s: {err}", flush=True)
            time.sleep(wait)
