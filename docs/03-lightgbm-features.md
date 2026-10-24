# 03 · LightGBM with weather, calendar and holiday features

## What this step does, in one sentence

It trains a machine-learning model that knows *what day it is* (weekday, holiday) and *what
the weather will be*, and measures how much that beats the step 2 baselines in the same exam.

## Why it matters

Step 2 showed where the simple models fail:

- **France in winter**: demand follows temperature (electric heating), and the baselines can't see temperature.
- **Germany in April–May**: Easter, 1 May, Ascension, Whit Monday. The baselines don't know about holidays.
- **Clock changes**: on the UTC grid, people's routines jump by an hour twice a year.
- **Price at 17–18 UTC**: the evening ramp, when solar fades and gas plants set the price.

Each of these points to information the model needs. Step 3 gives it that information, then
checks honestly how much each piece helps.

## Spec

### The model

**LightGBM**: a "gradient-boosted trees" model. It builds hundreds of small decision trees, each
fixing the previous ones' mistakes. It's the standard tool for tabular forecasting in industry
and in Kaggle competitions, because it handles non-linear effects on its own. For example,
demand rises when it's very cold *and* when it's very hot, with a flat middle.

- **One model per country × target** (8 models). Each one forecasts all 24 hours, with the
  hour as an input, so it learns from 24× more examples than one model per hour would.
- **Retrained every 7 days** in the backtest, on all data available at that moment (expanding window).
- **Same exam as step 2:** same origins (10:00 UTC), same test year, same scoring, and the same
  empirical intervals from past errors, so the comparison is apples to apples. LightGBM's own
  quantile models come in step 5.

### Features: what the model gets for each hour it forecasts

Every feature must be **known at 10:00 the day before**. That's the one rule.

| Group | Features | Why |
|---|---|---|
| **Calendar** | local hour, day of week, day of year, month | Routines follow the local clock, which fixes the clock-change problem |
| **Holidays** | public holiday (national, via the `holidays` library), bridge day (a workday squeezed between a holiday and a weekend), Christmas period (24 Dec–1 Jan) | The April–May and December errors |
| **Lags** | same hour 2 days before, 7 days before, 14 days before | The patterns the baselines used, now as inputs. All ≥ 48 h old, so always known at 10:00 |
| **Recent level** | last known value (09:00), mean of the last 24 h before the origin | "Where are we right now?" |
| **Weather** | temperature, humidity, cloud cover, wind at 100 m, solar radiation, plus temperature averaged over the past 24 h | Heating and cooling, solar and wind output. The 24 h average captures that buildings cool down slowly |

Price models get the same features with price lags instead of load lags. We don't use the
renewable output actuals: they're only known after the fact.

### Weather honesty: three versions of the model

| Version | Weather in training | Weather when forecasting the test year | Purpose |
|---|---|---|---|
| `lgbm_noweather` | none | none | What calendar and holidays alone buy |
| `lgbm_oracle` | observed | **observed** (cheating) | Upper bound: the score with perfect weather knowledge |
| `lgbm` | observed | **forecast issued ≥ 2 days ahead** | The honest model, the headline number |

"Trained on observed weather, run on forecast weather" is how this is done in industry. Archived
forecasts only exist from March 2024, too little history to train on alone.

The gap between `lgbm_oracle` and `lgbm` is the **cost of weather uncertainty**. Few student
projects measure it, and it's one result a recruiter will remember.

### Tuning without peeking

A handful of settings (number of trees, tree size, learning rate, L1 vs L2 loss) are chosen on a
**validation period, Jan–Jul 2025**, which is *before* the test year. The test year is run once,
with the chosen settings. Tuning on the test year would make the score look better than reality.

### Training windows

- **Load:** from 2021.
- **Price:** from 2023. The 2022 energy crisis (prices up to €3,000/MWh) would teach the model a
  market that no longer exists. We check this choice in validation (2021 vs 2023 start) rather than assume it.

### What we report

1. The step 2 table with the three LightGBM versions added (skill vs `naive_weekly`).
2. **Holiday vs normal days**, as promised in step 2: MAE on each.
3. **Feature importance**: which inputs the model leans on, per country. We expect temperature
   in France and holidays in Germany. If the data says otherwise, that's a finding too.
4. Error by month and hour, as in step 2: did the April–May and 17–18 UTC problems shrink?

### Code shape

```
src/energy_forecast/
  sources/open_meteo.py    + fetch_weather_forecast(): archived forecasts (previous-runs API)
  pipeline.py              + weather forecasts stored as extra columns (`*_fc`)
  features.py              build_features(): one row per target hour, every column known at 10:00 the day before
  models/lgbm.py           the three LightGBM versions
  backtest.py              models now also receive `future`: the columns allowed for tomorrow
```

**One interface change:** today a model gets the past of one series. LightGBM also needs
tomorrow's *allowed* inputs (forecast weather, calendar). So `predict(history, future)` gets:
- `history`: all columns, cut at 09:00 (as before)
- `future`: tomorrow's 24 hours, containing **only** the forecast-weather columns (or observed
  weather for the oracle, which is named as such)

Baselines simply ignore `future`. The leakage test grows: every column after the origin is
poisoned except those allowed in `future`, and no forecast may change.

**New dependencies:** `lightgbm`, `holidays`.

### Out of scope

Quantile LightGBM and conformal intervals (step 5), Chronos (step 4), regional holidays and
school holidays, renewable generation forecasts as inputs.

## Review

1. **The forecast timing trap.** Open-Meteo's `previous_day1` forecast is issued about 24 h
   before each hour. For 23:00 tomorrow that's 23:00 today, *after* our 10:00 deadline: still a
   leak. `previous_day2` is always issued before 10:00 the day before. *Decided:* use
   `previous_day2`. It's slightly older than what a real forecaster would have (about 48 h ahead
   instead of 14–37 h), so the honest model is a little pessimistic. Pessimistic beats leaky.
2. **Observed vs forecast weather mismatch.** The model learns on observed weather, then is fed
   slightly wrong (forecast) weather. That can hurt more than expected if it relied on fine
   detail. That's exactly what `lgbm_oracle` vs `lgbm` measures, so it's reported, not hidden.
3. **Forecast coverage.** Checked: complete from March 2024 to today, partial in Feb 2024, empty
   before. The test year (from Oct 2025) and its warm-up are fully covered. To verify in
   implementation: the same holds for all 19 cities, not just Berlin.
4. **Germany has regional holidays.** Some holidays only apply in some states (e.g. Corpus Christi
   in the south). National holidays only, for now. If April–June errors in DE stay high, a
   "share of population on holiday" feature is the next fix.
5. **Weekly retraining from scratch.** About 52 retrains × 8 models × 3 versions ≈ 1,250 fits.
   LightGBM trains on ~45k rows in seconds, so expect minutes, not hours. To measure.
6. **Building training rows without leaking.** For each past day we rebuild "what was known at
   10:00 the day before", with the same function used for forecasting. One function for both
   means training and forecasting can't drift apart. A test checks a training row equals the
   forecasting row for the same hour.
7. **Missing data.** LightGBM handles empty inputs natively, so the Spanish blackout day and the
   two-day-old lags around it don't need filling. Rows with no *target* are dropped from training.
8. **Overfitting holidays.** There are only ~10 holidays a year per country, so ~50 per country
   in training. Enough to learn "holiday = lower demand", too few for each holiday's own shape.
   That's why there's one `is_holiday` flag, not one flag per holiday.

## Implementation notes

- **Native LightGBM API** (`lgb.train`) instead of the scikit-learn wrapper: same model, no
  scikit-learn dependency.
- **Two layers of leakage tests**, each checked by deliberately breaking the code (adding a
  24 h lag, which isn't known at 10:00 for late hours):
  - the *poison* test (everything after 10:00 replaced by 10¹², forecasts must not change) still
    passed, because the backtest cut the data and the bad lag only found empty values;
  - the *training row = forecasting row* test failed, as it should: the model would have trained
    on a lag it never gets in real life.
  So the first test guards the backtest, the second guards the features.
- **Speed.** Tuning took 77 min. Profiling showed one fit takes ~0.7 s alone; the slowdown came
  from running 16 processes × 2 LightGBM threads on what behaves like ~8 real cores. With one
  thread per process, the full test-year run (24 jobs, ~1,460 fits) took **9 min**.

## Tuning (validation period, Jan–Jul 2025)

Full table: [`results/tuning.csv`](../results/tuning.csv). Scores are MAE relative to the best
setting for each country, averaged (1.000 = best everywhere).

| Target | Winner | Score | Training from 2023 instead of 2021 |
|---|---|---|---|
| Load | L2 loss, 127 leaves, from 2021 | 1.008 | ~4% worse: more history helps |
| Price | L1 loss, 31 leaves, from 2023 | 1.010 | ~3–4% *better*: the 2022 crisis hurts |

All 8 settings land within 1–7% of each other, so the model isn't fragile to its settings. The
price result confirms the spec's guess with data: a model trained on 2022 learns a market that
no longer exists. L1 loss winning for price also makes sense: it's less pulled by spike hours.

## Results (test year Oct 2025 → Sep 2026)

Full tables in [`results/`](../results). Skill = % lower MAE than "same hour last week".

**Load**

| | DE | FR | ES | NL |
|---|---|---|---|---|
| `lgbm` (honest) skill | **+52%** | **+68%** | **+50%** | **+49%** |
| `lgbm` MAPE | 2.6% | 2.2% | 2.4% | 2.6% |
| `lgbm_oracle` skill | +52% | +69% | +51% | +50% |
| `lgbm_noweather` skill | +43% | +47% | +36% | +32% |
| `mstl` (best of step 2) | +20% | +49% | +15% | +11% |

**Price**

| | DE | FR | ES | NL |
|---|---|---|---|---|
| `lgbm` (honest) skill | **+41%** | **+34%** | **+36%** | **+42%** |
| `lgbm` MAE (€/MWh) | 22.1 | 23.6 | 18.0 | 19.4 |
| `lgbm_oracle` skill | +44% | +35% | +36% | +43% |
| `lgbm_noweather` skill | +24% | +26% | +24% | +23% |
| best of step 2 | +14% | +20% | +19% | +14% |

### What the numbers say

1. **LightGBM roughly halves the error of "last week"** on load in every country, and gets to
   2.2–2.6% MAPE. For scale: the official day-ahead load forecasts published by grid operators
   are usually in the 2–3% range. This is a sanity check, not a claim to beat them; their task
   and timing differ.
2. **The cost of not knowing tomorrow's weather is small: 0.4–6%.** (Corrected in step 5; first
   written as 1–4%, which missed German price at 5.7%.) The honest model is barely
   worse than the oracle (DE load 1,389 vs 1,371 MW). The two-day-ahead temperature forecast,
   averaged over 4–5 cities, misses by only ~0.6 °C, and errors at different cities partly cancel.
   France pays the most (3.5% on load), as expected for the most weather-sensitive country.
3. **Weather is worth far more than its uncertainty costs.** Going from no weather to forecast
   weather cuts load error by 16–40% and price error by 11–26%. France's load benefits the
   most (−40%): electric heating again.
4. **Holidays are fixed.** On German public holidays, "last week" missed by 10.5 GW on average,
   MSTL by 7.4 GW, and `lgbm` by 1.4 GW, the same as a normal day. Spring in Germany, the worst
   period in step 2, dropped from ~3.6 GW to ~1.4 GW MAE in April and May.
5. **What the model relies on** ([`results/feature_importance.csv`](../results/feature_importance.csv)):
   - Load in DE/ES/NL: mostly last week's and the week before's values (60% and 15–20% of gain).
   - Load in France: the last known value, the recent 24 h level and the 24 h temperature carry
     54% between them. French demand is driven by the current weather regime, not the weekly routine.
   - Price: wind at 100 m matters most in DE and NL (13–14%), where wind sets the price on many
     days, and much less in ES and FR (solar and nuclear markets).
   - *Caveat:* "gain" measures average usefulness over all hours. Holidays score low because
     they're rare, even though they matter enormously on the days they occur (point 4).
6. **The 90% ranges still under-cover (84–87%).** Same empirical method as step 2, same
   problem. This is step 5's job.
