# European Electricity Forecasting

Probabilistic forecasts of hourly electricity demand and day-ahead prices for Germany, France,
Spain and the Netherlands. Every model is judged with rolling-origin backtests and calibrated
prediction intervals, not a single train/test split.

> Status: data pipeline, baselines, LightGBM, calibrated intervals, significance tests and the live app done. Foundation model (Chronos) next.

## Results so far

Day-ahead forecasts issued at 10:00 UTC for all 24 hours of the next day, replayed every day for
a full test year (Oct 2025 → Sep 2026). The model only sees weather *forecasts* issued at least
two days ahead, as a real forecaster would. Skill = % lower error than "same hour last week",
with 95% block-bootstrap intervals.

| LightGBM (honest weather) | DE | FR | ES | NL |
|---|---|---|---|---|
| Load: skill | +52% [45, 57] | +68% [62, 73] | +50% [44, 54] | +49% [45, 54] |
| Load: MAPE | 2.6% | 2.2% | 2.4% | 2.6% |
| Price: skill | +41% [35, 45] | +34% [28, 40] | +36% [29, 43] | +42% [37, 47] |
| Price: MAE (€/MWh) | 22.1 | 23.6 | 18.0 | 19.4 |

- **The lead is real:** LightGBM beats every baseline in all 8 series (Diebold–Mariano, Holm-corrected p < 0.001).
- **Weather uncertainty costs only 0.4–6%** of error, and the gap is statistically detectable in just 3 of 8 series.
- **Holidays solved:** on German public holidays the error drops from 10.5 GW ("last week") to 1.4 GW.
- **Calibrated ranges:** adaptive conformal inference lifts 90%-range coverage from 84–87% to
  88–90% for load; conformalized quantile regression gives the sharpest well-calibrated price
  ranges. Plain quantile regression covers only 66–75%.

Details: [docs/03](docs/03-lightgbm-features.md#results-test-year-oct-2025--sep-2026), [docs/05](docs/05-intervals-significance.md#results-test-year-oct-2025--sep-2026).

## Live app

Tomorrow's forecast, published every day, plus a public track record and the backtest explorer:
**[app link: added after deployment]**. GitHub Actions runs the forecast at 12:00 and 18:00 UTC
([workflow](.github/workflows/live.yml)); the published files live on the
[`live` branch](../../tree/live). The free hosting sleeps after a few quiet days, so the first
visit may take ~30 s to wake it up.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
energy-forecast all          # fetch + build + validate (first run: ~5 min)
energy-forecast tune         # pick LightGBM settings on Jan–Jul 2025 (optional, slow)
energy-forecast backtest     # replay the test year for every model (~20 min)
energy-forecast evaluate     # score them → results/
energy-forecast intervals    # compare interval methods
energy-forecast significance # DM tests and skill error bars
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
5. ✅ Conformal intervals and significance tests
6. ✅ Live forecasts and Streamlit app

## Docs

- [How the code is organised](docs/00-code-guide.md)
- [01 · Data pipeline](docs/01-data-pipeline.md): spec, design decisions, review
- [02 · Baselines and backtesting](docs/02-baselines-backtesting.md): the forecasting task, the backtest, metrics
- [03 · LightGBM with features](docs/03-lightgbm-features.md): weather, calendar, holidays; honest vs oracle weather
- [05 · Intervals and significance](docs/05-intervals-significance.md): conformal calibration, Diebold–Mariano tests, error bars
- [06 · The app](docs/06-app.md): live daily forecasts, public track record, backtest explorer
