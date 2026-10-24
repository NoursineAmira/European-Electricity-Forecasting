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
    open_meteo.py            weather data
tests/test_pipeline.py       checks the tricky parts without touching the internet
docs/                        one file per project step: spec, decisions, review
```

## The flow, step by step

**1. `fetch`.** For every country and every 3-month period, call the three sources
(power, price, weather) and save what comes back to `data/raw/<country>/`, e.g.
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
```
