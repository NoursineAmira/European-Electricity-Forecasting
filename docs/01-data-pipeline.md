# 01 · Data pipeline

## What this step does, in one sentence

It downloads electricity and weather data for four countries and saves one clean hourly table
per country, which every later model reads.

## Why it matters

A forecast can only be as good as its data. If timestamps are shifted by an hour, or a missing
day gets silently filled with zeros, the models will learn the wrong thing and the results table
will mean nothing. So this step has one job: produce data we can trust.

## Spec

### Sources (both free, no API key)

| Source | What we take | Native resolution |
|---|---|---|
| [Energy-Charts](https://api.energy-charts.info) (Fraunhofer ISE) | Load (electricity demand), solar, wind onshore, wind offshore | 15 min (FR: 1 h) |
| Energy-Charts `/price` | Day-ahead price per bidding zone | 1 h, 15 min since Oct 2025 |
| [Open-Meteo archive](https://open-meteo.com/en/docs/historical-weather-api) | Temperature, humidity, cloud cover, wind at 100 m, solar radiation | 1 h |

Countries: Germany, France, Spain, Netherlands. History: 2021-01-01 to yesterday.
All of this is set in `config.yaml`, so adding a country means editing one file.

### Output

One file per country: `data/processed/<country>.parquet`. One row per hour.

| Column | Meaning |
|---|---|
| `timestamp` (index) | Start of the hour, **in UTC** |
| `load_mw` | Average electricity demand during that hour (the main target) |
| `price_eur_mwh` | Day-ahead price (second target) |
| `solar_mw`, `wind_onshore_mw`, `wind_offshore_mw` | Renewable generation |
| `temperature_2m`, `relative_humidity_2m`, `cloud_cover`, `wind_speed_100m`, `shortwave_radiation` | Weather, averaged over the country's big cities |

### Steps

1. **Fetch** (`energy-forecast fetch`): download each source in 3-month chunks and save each
   chunk untouched in `data/raw/`. If a chunk is already on disk, skip it. Re-running is cheap,
   and an internet hiccup only costs one chunk.
2. **Build** (`energy-forecast build`): read the raw chunks, convert to hourly, join everything
   on `timestamp`, write the processed file.
3. **Validate** (`energy-forecast validate`): print a short health report per country (row
   count, missing hours, gaps, value ranges).

### Rules

- **Everything in UTC.** Local time has daylight saving: one day a year has 23 hours, another
  has 25. UTC never jumps, so no hour is lost or counted twice.
- **15-min to hourly = average.** Four 15-min readings of demand in MW become their mean, which
  is the average power during the hour.
- **Missing stays missing.** Gaps are left as `NaN` (empty) and reported, never filled with zero.
  Filling is a modelling decision, made later and on purpose.
- **Negative prices are real.** On sunny, windy days prices drop below zero. Keep them.
- **Zero load is not real.** A country never uses 0 MW, so load ≤ 0 is a meter error and becomes
  missing. This happens *before* averaging, so one bad 15-min reading can't drag a whole hour down.
- The current (incomplete) quarter is always re-downloaded, so the data stays fresh.

### Out of scope here

Features (lags, holidays, calendar), models, and plots. Those come in later steps.

## Review

Problems I checked the spec for before writing code:

1. **Leakage (most important).** To forecast tomorrow's demand you don't know tomorrow's *actual*
   weather, only a *forecast* of it. Training with actual weather makes the model look better
   than it could ever be in real life. *Decision:* store actual weather now (simple, complete
   since 2021), and say clearly in the README that the first models use it. In a later step,
   swap in Open-Meteo's archived forecasts (available from 2022), and show how much accuracy
   drops. That comparison is itself a strong result to show recruiters.
   The same applies to solar and wind *actuals*: they're inputs for analysis, not for the
   day-ahead model.
2. **Licence.** Spanish prices are marked "private and internal use only". Data is never
   committed (`data/` is in `.gitignore`), so anyone who runs the repo downloads their own.
3. **Price resolution change.** Prices moved from hourly to 15-min in Oct 2025. Averaging to
   hourly handles both periods the same way.
4. **Timestamps at chunk edges.** Chunks are requested on day boundaries and duplicates are
   dropped after joining, so an hour sitting on a boundary is never counted twice.
5. **API politeness.** Energy-Charts is a public research service: one request at a time,
   a short pause between calls, and retries with backoff on errors.
6. **Weather averaging is unweighted.** A mean over 4–5 cities is a simplification. Good enough
   to start; population weights could come later if the error analysis shows weather matters.

## What the first run found

All four countries: 50,928 hours (2021-01-01 → yesterday), no broken hourly grid.

| Finding | Verdict |
|---|---|
| Last 2 hours of yesterday missing everywhere | Not published yet. Harmless, refilled on the next run. |
| Spain, 28 Apr 2025, ~12 h of load missing | The real **Iberian blackout**. Kept as a gap; backtests must skip or flag that day. |
| Spain, 17 Mar 2026, load = 0 MW | Meter error. Led to the "zero load" rule above. |
| France, offshore wind 42% missing | French offshore farms only came online in late 2022. Real. |
| France, price peak €2,988/MWh | Real: April 2022 energy-crisis spike. |
| Netherlands solar ≈ 60 MW on average | Source covers only part of Dutch solar (true capacity is 20+ GW). **Don't use NL solar.** |
