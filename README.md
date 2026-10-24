# European Electricity Forecasting

Probabilistic forecasts of hourly electricity demand and day-ahead prices for Germany, France,
Spain and the Netherlands. Every model is judged with rolling-origin backtests and calibrated
prediction intervals, not a single train/test split.

> Status: data pipeline and baselines done. Feature-based models in progress.

## Results so far

Day-ahead forecasts issued at 10:00 UTC for all 24 hours of the next day, replayed every day for
a full test year (Oct 2025 → Sep 2026). Skill = % lower error than "same hour last week".

| Load, best baseline (MSTL) | DE | FR | ES | NL |
|---|---|---|---|---|
| Skill vs last week | +20% | +49% | +15% | +11% |
| MAPE | 4.4% | 3.6% | 4.2% | 4.6% |

Prices are harder (best skill 14–20%), and the baseline 90% intervals under-cover (79–87%).
Both are targets for the next steps. Details: [docs/02](docs/02-baselines-backtesting.md#results).

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
energy-forecast all          # fetch + build + validate (first run: ~5 min)
energy-forecast backtest     # replay the test year for every model (~10 min)
energy-forecast evaluate     # score them → results/
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
2. ✅ Baselines (seasonal naive, MSTL) and rolling-origin backtest
3. LightGBM with lag and calendar features
4. Zero-shot foundation model (Chronos)
5. Conformal intervals and significance tests
6. Streamlit app

## Docs

- [How the code is organised](docs/00-code-guide.md)
- [01 · Data pipeline](docs/01-data-pipeline.md): spec, design decisions, review
- [02 · Baselines and backtesting](docs/02-baselines-backtesting.md): the forecasting task, the backtest, metrics
