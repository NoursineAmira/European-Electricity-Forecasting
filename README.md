# European Electricity Forecasting

Probabilistic forecasts of hourly electricity demand and day-ahead prices for Germany, France,
Spain and the Netherlands. Every model is judged with rolling-origin backtests and calibrated
prediction intervals, not a single train/test split.

> Status: data pipeline, baselines and LightGBM done. Foundation model and calibrated intervals next.

## Results so far

Day-ahead forecasts issued at 10:00 UTC for all 24 hours of the next day, replayed every day for
a full test year (Oct 2025 → Sep 2026). The model only sees weather *forecasts* issued at least
two days ahead, as a real forecaster would. Skill = % lower error than "same hour last week".

| LightGBM (honest weather) | DE | FR | ES | NL |
|---|---|---|---|---|
| Load: skill | +52% | +68% | +50% | +49% |
| Load: MAPE | 2.6% | 2.2% | 2.4% | 2.6% |
| Price: skill | +41% | +34% | +36% | +42% |
| Price: MAE (€/MWh) | 22.1 | 23.6 | 18.0 | 19.4 |

- **Weather uncertainty costs only 1–4%:** using the forecasts actually available instead of the
  weather that really happened (an "oracle") barely changes the score.
- **Holidays solved:** on German public holidays the error drops from 10.5 GW ("last week") to 1.4 GW.
- **Intervals still under-cover** (84–87% for a 90% range): the target of step 5.

Details: [docs/03](docs/03-lightgbm-features.md#results-test-year-oct-2025--sep-2026), [docs/02](docs/02-baselines-backtesting.md#results).

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
energy-forecast all          # fetch + build + validate (first run: ~5 min)
energy-forecast tune         # pick LightGBM settings on Jan–Jul 2025 (optional, slow)
energy-forecast backtest     # replay the test year for every model (~20 min)
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
3. ✅ LightGBM with weather, calendar and holiday features
4. Zero-shot foundation model (Chronos)
5. Conformal intervals and significance tests
6. Streamlit app

## Docs

- [How the code is organised](docs/00-code-guide.md)
- [01 · Data pipeline](docs/01-data-pipeline.md): spec, design decisions, review
- [02 · Baselines and backtesting](docs/02-baselines-backtesting.md): the forecasting task, the backtest, metrics
- [03 · LightGBM with features](docs/03-lightgbm-features.md): weather, calendar, holidays; honest vs oracle weather
