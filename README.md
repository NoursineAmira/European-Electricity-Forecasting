# European Electricity Forecasting

Probabilistic forecasts of hourly electricity demand and day-ahead prices for Germany, France,
Spain and the Netherlands. Every model is judged with rolling-origin backtests and calibrated
prediction intervals, not a single train/test split.

> Status: data pipeline done. Models in progress.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
energy-forecast all          # fetch + build + validate (first run: ~5 min)
pytest
```

Output: `data/processed/<country>.parquet`, one row per UTC hour with load, price,
renewables and weather.

## Data

| Source | Data | Licence |
|---|---|---|
| [Energy-Charts](https://api.energy-charts.info) (Fraunhofer ISE) | Load, solar, wind, day-ahead prices | Mostly CC BY 4.0; ES prices private use only |
| [Open-Meteo](https://open-meteo.com) | Historical weather | CC BY 4.0 |

Data is not committed. Running the pipeline downloads your own copy.

**Known limitation:** the first models use *observed* weather, which a real forecaster
wouldn't have yet. A later step swaps in archived weather *forecasts* and measures the gap.
See [docs/01-data-pipeline.md](docs/01-data-pipeline.md).

## Roadmap

1. ✅ Data pipeline
2. Baselines: seasonal naive, ETS
3. LightGBM with lag and calendar features
4. Zero-shot foundation model (Chronos)
5. Rolling-origin backtests, quantile and conformal intervals
6. Streamlit app

## Docs

- [How the code is organised](docs/00-code-guide.md)
- [01 · Data pipeline](docs/01-data-pipeline.md): spec, design decisions, review
