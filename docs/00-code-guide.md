# How the code is organised

A map of the repo in plain words. Read this first.

```
config.yaml                  what to download: countries, cities, dates
src/energy_forecast/
  config.py                  reads config.yaml into a Python object
  cli.py                     the `energy-forecast` command
  pipeline.py                fetch → build → validate
  sources/
    http.py                  one function: download JSON, retry if the network fails
    energy_charts.py         electricity data (load, solar, wind, prices)
    open_meteo.py            weather: observed, and forecasts as they were issued
  features.py                what the model is told about each hour (calendar, holidays, lags, weather)
  models/
    base.py                  the contract every model follows
    baselines.py             the simple models every later model must beat
    lgbm.py                  LightGBM, in three weather versions
  backtest.py                the exam: replays the past one day at a time
  metrics.py                 how forecasts are scored
  experiment.py              runs every model on every country, then scores them
tests/                       checks the tricky parts without touching the internet
docs/                        one file per project step: spec, decisions, review
results/                     score tables (committed; small and derived)
```

## The flow, step by step

**1. `fetch`.** For every country and every 3-month period, call the four sources
(power, price, observed weather, weather forecasts from 2024 on) and save what comes back to `data/raw/<country>/`, e.g.
`data/raw/de/power_2024Q3.parquet`. If the file already exists, skip it. That's why a second run
is instant. The most recent quarter is always re-downloaded because it's still growing.

**2. `build`.** For each country, glue all the raw files together and pass each source through
`to_hourly`, then put the three side by side in one table. Saved to `data/processed/<country>.parquet`.

**3. `validate`.** For each column: how much is missing, the longest run of missing hours, and
min/mean/max. This is where you'd spot a broken source (e.g. load suddenly at 0).

## The two functions worth understanding

**`quarters(start, end)`** cuts a long date range into calendar quarters
(Jan–Mar, Apr–Jun, ...). Smaller requests are kinder to the free APIs, and if one fails you only
redo that piece.

**`to_hourly(df, start, end)`** is where data quality is decided:

1. drop duplicate timestamps (an hour can appear at the edge of two chunks)
2. average 15-minute values into hours (`resample("h").mean()`)
3. place everything on a complete hourly grid from `start` to `end`, so a missing hour becomes
   an empty cell you can see, instead of a row that silently isn't there

## Models and the backtest (step 2)

**A model is anything with a `predict(history, future, horizon)` method.** It returns 24
numbers. That's the whole contract, in `models/base.py`. A one-line rule and LightGBM plug in
the same way.

- `history`: every column, up to 09:00 today
- `future`: from 10:00 today to the end of tomorrow, but **only the columns the model declared
  in `future_columns`**. For the honest LightGBM that's the weather *forecasts* (`*_fc`).
  Baselines declare none.
- `horizon`: the 24 hours to forecast

**`backtest.run`** is the exam. For each day it:
1. cuts the data at 09:00 (`data.loc[: origin - 1h]`): the model *physically* can't see later data
2. builds `future` with only the allowed columns
3. asks the model for tomorrow's 24 hours and stores them next to what really happened

**`backtest.add_quantiles`** turns one number into a range. For each hour of the day, it
looks at the model's misses over the last 8 weeks, e.g. "at 18:00 we usually miss by −1 to
+2 GW", and adds them to tomorrow's forecast. The tricky bit is `shift(2)`: when forecasting
day T we're standing on day T−1 at 10:00, so the last *complete* day of errors is T−2.
A test checks this.

**`experiment.evaluate`** only scores hours where *every* model produced a forecast and the
actual is known. Otherwise a model could look better just by skipping hard hours.

## LightGBM and its features (step 3)

**`features.build_features`** answers "what did we know about this hour at 10:00 the day
before?" One row per hour, with:
- **calendar** in *local* time: hour, weekday, day of year, public holiday, bridge day, Christmas week
- **lags**: the value 2, 7 and 14 days earlier (all older than 38 h, so always known)
- **recent level**: the last known value (09:00) and the mean of the 24 h before it
- **weather** at that hour, plus the temperature averaged over the past 24 h

The same function builds the training rows and the forecasting rows. That's on purpose: if they
were two functions, one could quietly use something the other can't, and the model would learn
from inputs it never gets in real life. `test_training_row_equals_forecasting_row` checks it.

**`models/lgbm.py`** has one class and three versions, which only differ in tomorrow's weather:

| Version | Weather for tomorrow |
|---|---|
| `lgbm_noweather` | none |
| `lgbm_oracle` | what really happened (cheating, on purpose: an upper bound) |
| `lgbm` | the forecast issued two days before (honest) |

All three *train* on observed weather. Every 7 days the model retrains on everything before
the current origin.

**Tuning (`energy-forecast tune`)** tries every combination in `config.yaml → lgbm.tuning.grid`
on Jan–Jul 2025, before the test year. The winning settings are copied into `lgbm.targets` by
hand, so `config.yaml` is the record of what was chosen.

## Why parquet and not CSV?

Parquet keeps column types (a timestamp stays a timestamp with its time zone), is about 10x
smaller, and loads much faster. Open one in Python with `pd.read_parquet(path)`.

## Why UTC?

In local time, one night in March has no 2 a.m. and one night in October has two. In UTC every
day has exactly 24 hours, so the hourly grid never breaks. Convert to local time only when you
need it as a feature (people's habits follow local clocks).

## Commands

```bash
energy-forecast fetch --countries de fr   # only some countries
energy-forecast build
energy-forecast validate
energy-forecast all                       # all three in order
energy-forecast tune                      # pick LightGBM settings on the validation period
energy-forecast backtest --models naive_weekly mstl --targets load_mw
energy-forecast evaluate                  # writes results/*.csv and prints the table
```
