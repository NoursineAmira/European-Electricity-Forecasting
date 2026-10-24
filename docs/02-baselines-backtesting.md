# 02 · Baselines and backtesting

## What this step does, in one sentence

It builds a fair "exam" that every model will sit, and gets the first scores from simple models
that any fancier model must beat.

## Why it matters

A number like "MAE = 1,200 MW" means nothing alone. Is that good? It's only meaningful next to
what a dumb rule gets ("tomorrow looks like the same day last week"). And the exam has to be
honest: the model may only use information that would really have been available when the
forecast was made. Most student projects get this wrong. Getting it right is the point of the
project.

## Spec

### The forecasting task

We copy the real day-ahead electricity market, simplified to UTC:

- **When:** every day at **10:00 UTC** (the auction closes at 12:00 Brussels time).
- **What:** all **24 hours of the next day** (00:00–23:00 UTC). Lead times run from 14 to 37 hours.
- **Known at that moment:** actuals up to 09:00 UTC today. Nothing after.
- **Targets:** `load_mw` (main) and `price_eur_mwh`, for all four countries.
- **Output:** a median forecast plus quantiles `0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95`.
  So every model says "most likely X" and also "90% sure it's between A and B".

### The exam: rolling-origin backtest

Instead of one train/test split, we replay the past one day at a time:

```
day 1:  train on everything up to 09:00 → forecast tomorrow → compare with what happened
day 2:  move forward one day, repeat
...
```

- **Test period:** 2025-10-01 → 2026-09-30. One full year, so every season is scored once.
- **Warm-up:** origins start 8 weeks earlier (2025-08-06). Those weeks are used to calibrate
  the intervals, not scored.
- **Training data:** everything before the origin (expanding window); each model decides how
  much of it to use.
- **Missing actuals:** hours with no actual value are skipped in scoring, and the count is reported.

### Baseline models

| Name | Rule, in plain words |
|---|---|
| `naive_daily` | Same hour, two days ago (the latest same hour already known at 10:00) |
| `naive_weekly` | Same hour, same weekday, last week |
| `weekly_mean` | Average of the same hour on the same weekday over the last 4 weeks |
| `mstl` | Splits the series into a daily pattern + a weekly pattern + a trend, and forecasts each (statsforecast `MSTL` with `AutoETS` trend). Refit every day on the last 8 weeks. |

**Intervals for all baselines:** look at the model's own errors over the previous 8 weeks, for
the same lead hour, and use their quantiles. Example: if 90% of past errors fell between
−2 GW and +3 GW, tomorrow's 90% interval is median −2 GW to median +3 GW.

### Scoring

| Metric | Plain meaning | Used for |
|---|---|---|
| MAE | Average size of the miss, in MW or €/MWh | both targets |
| RMSE | Like MAE, but punishes big misses more | both targets |
| MAPE | Average miss in % | load only (prices can be 0 or negative, which breaks %) |
| Skill vs `naive_weekly` | `1 − MAE_model / MAE_naive_weekly`. 0.2 means "20% better than the dumb rule" | headline number |
| Pinball loss | Scores the quantiles: rewards intervals that are narrow *and* right | probabilistic quality |
| Coverage (90%) | How often the truth landed inside the 90% interval. Should be ≈ 90% | calibration |

Results are also broken down by **hour of day** and **month**, so we see *where* each model
fails, not just how much. Holidays are scored like any other day; a holiday breakdown comes with
the holiday features in step 3.

### Code shape

```
src/energy_forecast/
  backtest.py      the exam: loops over origins, hides the future, stores forecasts
  metrics.py       the scoring functions above
  models/
    base.py        what every model must provide: fit(history) and predict(history, origin)
    baselines.py   the four models
```

- Command: `energy-forecast backtest --models naive_weekly mstl --targets load_mw`
- Forecasts saved to `data/backtests/<country>/<target>/<model>.parquet` (one row per origin × hour)
- `energy-forecast evaluate` writes `results/metrics.csv` and prints the comparison table. Only
  scores are committed, not data.

**Leakage guard:** `backtest.py` cuts the history at the origin *before* the model sees it. A test
passes a history whose future is filled with an impossible value and checks it never leaks into
a forecast. Models can't cheat even by accident.

### Out of scope

Weather, holidays and calendar features (step 3), foundation models (step 4), conformal
intervals and significance tests (step 5).

## Review

1. **UTC day vs market day.** The real market trades the *local* (Brussels) day. Using UTC
   shifts it by 1–2 hours. *Accepted:* the task stays just as hard and all models face the same
   rule. Stated in the README.
2. **Daylight saving and weekly lags.** On our UTC grid, people's habits jump by one hour when
   clocks change. In the week after each change, `naive_weekly` compares hours shifted by one.
   Small, real, and visible in the by-month breakdown, so we report it instead of hiding it.
   Calendar features in step 3 fix this properly.
3. **Price is harder than load.** Spikes and negative prices make price errors heavy-tailed.
   MAPE is useless there (see table), and RMSE will be dominated by a few spike hours. MAE and
   skill score are the headline numbers for price.
4. **Is 2022 a problem?** The energy crisis (prices up to €3,000/MWh) sits in the training data.
   The baselines only look back 2–8 weeks, so it doesn't affect them. Step 3 has to decide
   whether to down-weight or drop it.
5. **Interval warm-up.** Empirical intervals need past errors before they work. The 8-week
   warm-up before the test year solves this; no scored day lacks calibration data.
6. **Runtime.** About 425 origins × 4 countries × 2 targets. Naive models are instant. One MSTL
   fit takes ~0.3 s (measured), so refitting *every day* is ~17 min serial, ~3 min with the 8
   series run in parallel. Daily refits are simpler and more honest than weekly ones, so we do that.
7. **Libraries.** statsforecast, statsmodels, LightGBM and matplotlib all have wheels for
   Python 3.14 (checked). No blocker.
8. **One test year is one sample.** A model can win one year by luck. Step 5 adds a
   Diebold–Mariano test (a standard check that the gap between two forecasters isn't noise).
   For now, scores are reported per country so the results aren't one lucky number.
9. **Holidays.** *Decided:* no special handling in this step. They stay in the score (a real
   forecaster can't skip Christmas), and the holiday breakdown moves to step 3.

## Results

Test year 2025-10-01 → 2026-09-30, ~8,760 scored hours per series. Full tables in
[`results/`](../results). **Skill** = % better than "same hour last week" (higher is better).

**Load: skill vs `naive_weekly`**

| | DE | FR | ES | NL |
|---|---|---|---|---|
| `mstl` | **+20%** | **+49%** | **+15%** | **+11%** |
| `weekly_mean` | −7% | −9% | −21% | −2% |
| `naive_daily` | −116% | −12% | −122% | −57% |
| MAE of best (MW) | 2,301 | 1,795 | 1,144 | 611 |
| MAPE of best | 4.4% | 3.6% | 4.2% | 4.6% |

**Price: skill vs `naive_weekly`**

| | DE | FR | ES | NL |
|---|---|---|---|---|
| `mstl` | +6% | **+20%** | **+19%** | +4% |
| `weekly_mean` | **+14%** | +7% | +3% | **+14%** |
| MAE of best (€/MWh) | 32.1 | 28.9 | 22.5 | 29.0 |

### What the numbers say

1. **MSTL wins load everywhere.** It learns the daily and weekly rhythm *and* follows the
   recent level, while "last week" can't react when the weather turns.
2. **France is the outlier (+49%).** Many French homes heat with electricity, so French demand
   swings with temperature far more than elsewhere. "Last week" misses by 5–7 GW in winter
   months, MSTL by about 2.5 GW. *Implication:* weather features should help France most in step 3.
3. **`naive_daily` is terrible for load** (−116% in DE): two days before a Monday is a Saturday.
   The weekly pattern dominates demand.
4. **April–May is hard for everyone in Germany** (MAE ~3.6 GW, worst months). That's Easter,
   1 May, Ascension and Whit Monday. These holidays are why step 3 adds a holiday calendar.
5. **Price is much harder.** Best skill is only 14–20%. Errors peak at 17–18 UTC, the evening
   ramp when solar drops out and gas plants set the price. Renewable and weather inputs should matter.
6. **The 90% ranges are too narrow: actual coverage is 79–87%.** The errors of the past 8 weeks
   underestimate tomorrow's uncertainty, especially when seasons change. This is the honest
   starting point for step 5 (conformal calibration).
7. **Runtime was ~10.5 min, not ~3** (16 cores). Each MSTL fit was slower inside parallel
   workers than measured alone, probably because the workers compete for the same CPU threads.
   Fine for now; worth revisiting if step 3 multiplies the number of fits.
