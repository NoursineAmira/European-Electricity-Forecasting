# 06 · The app: live forecasts and a public track record

## What this step does, in one sentence

It publishes a small website that shows tomorrow's forecast every day, keeps score of how those
live forecasts actually did, and lets anyone explore the backtest results.

## Why it matters

A recruiter gives a GitHub repo about 30 seconds. A table in a README says "trust me"; a page
with **today's forecast and a running score against reality** shows it. A live track record is
also the only evidence that can't be accidentally tuned: those forecasts were published
before the truth existed.

## Spec

### Three moving parts

```
GitHub Actions (daily, 12:00 UTC)        `live` branch                   Streamlit app
  fetch new data                  ──►    forecast_latest.csv     ──►     reads the CSVs
  replay last 120 days                   history.csv                     (no models, no heavy deps)
  publish tomorrow + fill actuals
```

1. **A daily job** runs the existing pipeline on GitHub's servers: fetch the new data,
   forecast tomorrow, and fill in yesterday's actuals for past forecasts.
2. **A `live` branch** stores two small CSV files. It's separate from `main`, so daily updates
   don't clutter the project's history.
3. **The app** only *reads* those files and the committed `results/` tables. It never trains a
   model, so it stays fast and cheap to host.

### The daily job (`energy-forecast live`)

It reuses the backtest code: no new modelling.

1. `fetch` + `build`, with the raw data cached between runs (only new quarters are downloaded).
2. For each published series, run the backtest over the **last 120 days up to today**: the
   same models, same weekly retraining, same 10:00 issue time and same data cut at 09:00.
3. Add the ranges with the method chosen in step 5: **ACI for load, CQR for price**.
4. Keep today's issue, i.e. tomorrow's 24 hours, and append it to `history.csv`. Rows from
   earlier days get their `actual` filled in as the truth arrives.
5. Force-push both files to the `live` branch as a single commit.

Why replay 120 days instead of keeping state between runs? There's nothing to corrupt: every
run rebuilds everything from public data, so a failed or skipped day fixes itself the next day.

**Running at 12:00 UTC, not 10:00:** measured on 2026-10-24, German load appears about 1.75 h
after real time. At 12:00 the data up to the 09:00 cut-off has safely arrived. The forecast
still only *uses* data up to 09:00, so this changes nothing about honesty; the app states both
times.

### What's published

| Series | Load | Price |
|---|---|---|
| Germany, France, Netherlands | ✅ | ✅ |
| Spain | ✅ | ❌ (licence; review point 1) |

### The app: four pages

| Page | Shows |
|---|---|
| **Tomorrow** | For a chosen country: tomorrow's load (and price) as a line with 50% and 90% bands, today's actuals so far for context, in **local time**. Header: "Issued 12:00 UTC using data up to 09:00 UTC on …". |
| **Track record** | Every published forecast against what happened: MAPE / MAE, 90% coverage, a chart of the daily error. Grows each day. Until 7 days exist it says so plainly instead of showing a misleading number. |
| **Backtest** | The test-year results from `results/`: skill with error bars, error by month and hour, holidays vs normal days, interval methods and coverage by month. Selectors for country and target. |
| **How it works** | A one-screen explanation in plain words, with links to `docs/`. |

Charts with Altair (ships with Streamlit, no extra dependency), following the repo's
`dataviz` guidance for colours and readability in light and dark mode.

### Code shape

```
src/energy_forecast/live.py      run_live(): replay, intervals, append to history
.github/workflows/live.yml       daily schedule, cache data/raw, push to the `live` branch
app/streamlit_app.py             the four pages
app/requirements.txt             streamlit, pandas, altair (not the forecasting package)
config.yaml                      + live: replay_days, published series, interval method per target
```

- The app reads the `live` files from `raw.githubusercontent.com` (cached for an hour), or
  from a local folder when run with `streamlit run app/streamlit_app.py` for development.
- Hosting: **Streamlit Community Cloud** (free, deploys straight from the GitHub repo).

### Out of scope

User accounts, choosing your own model, alerts, mobile layout beyond Streamlit's default,
Chronos (step 4).

## Review

1. **Spanish prices: licence.** Energy-Charts marks Spanish prices "private and internal use
   only … whether in raw or derived form". A public forecast of them is a derived form, so
   Spanish price is **not published** in the app or on the `live` branch.
   **Open question for you:** `results/` already contains *aggregate scores* for Spanish price
   (e.g. MAE 18.0 €/MWh), committed in steps 2–5. A strict reading of "derived form" could
   include those. Options: (a) keep them (aggregate error statistics, no prices, a common
   reading), or (b) drop ES price from the committed results and README. I lean (a), but it's
   your repo and your name on it, so it's your call.
2. **Data still arriving.** Measured delay is ~1.75 h for German load; the other countries
   still need checking during implementation. If a country's 09:00 value is regularly missing
   at 12:00, its `last_value` feature would be empty in live runs but never was in the backtest.
   *Decided:* the job logs any missing 09:00 value, and the app shows a "data incomplete"
   note for that series rather than hiding it.
3. **Observed weather gets revised.** For the last few days, the Open-Meteo archive holds model
   estimates that are later replaced by reanalysis. Live runs therefore train on slightly
   different recent weather than a re-run a week later would. Effect: tiny (a few days out of
   five years of training). Documented, not engineered around.
4. **Approximate thermostat.** ACI's thermostat needs a long run to settle; a 120-day replay
   gives it ~64 days after the 8-week warm-up. Live ranges will behave closer to `empirical`
   than the backtest's ACI did. Stated on the Track record page. If live coverage ends up
   clearly below 90%, the fix is a longer replay window, which costs runtime.
5. **Runtime on GitHub Actions.** Per run: load uses `lgbm` (~17 retrains + 120 forecasts) and
   price uses `lgbm_qr` (~17 × 7 = 119 fits) per country. On GitHub's 2–4-core runners, expect
   ~20–30 min/day. Free for public repos. The job has a 60-minute timeout. If a run fails, the
   app keeps showing the last good forecast **with its date**, never as if it were today's.
6. **Permissions.** The workflow pushes to `live` with GitHub's built-in token (`contents:
   write`, set in the workflow file). Pushing the workflow file itself from your machine needs
   your git credential to have the `workflow` scope. If the push is rejected, you add it once
   in GitHub's token settings (I'll give exact steps; I can't change account settings for you).
7. **Things only you can do.** Creating the Streamlit Community Cloud app means signing in with
   your GitHub account and clicking "Deploy". I'll prepare everything and give you the exact
   clicks. Free apps sleep after a few days without visitors and take ~30 s to wake up: worth
   a note in the README so a recruiter isn't confused.
8. **Honest labelling.** Backtest numbers and live numbers are never mixed on one chart or in one
   table. Every page says which it is, and the live track record starts at launch with no
   backfilled history: backfilling would turn a live record back into a backtest.
9. **Timezones.** Everything stays UTC inside; the app converts to each country's local time for
   display, since "tomorrow 18:00" means local time to a reader.

## Implementation notes

**Data delays, measured** (review point 2). On Saturday evening German load was ~1.75 h behind
real time. On Sunday morning (also the clock-change day) all four countries were **6+ hours**
behind: at 09:12 UTC the latest load was 02:30–03:15. A single 12:00 run can't count on having
the 09:00 value. *Changed:* the job runs at **12:00 and 18:00 UTC**; the second run replaces
the first. Both still only use data up to 09:00, so the later run isn't "peeking": it just has
a better chance that all of that data has arrived. Each forecast carries a `data_complete`
flag and the app shows a note when it's false. Day-ahead prices are published a day ahead, so
price is always complete.

**Three files instead of two.** The `live` branch holds:
- `history.csv`: every published forecast (24 hours × series × day), with `actual` filled in as it arrives
- `recent.csv`: the last ~3 days of actual values, for context on the Tomorrow chart
- `status.json`: when the run happened, the data cut, and `data_complete` per series

The spec's `forecast_latest.csv` would have duplicated rows already in `history.csv`.

**The weather archive has no future dates.** Asking it for tomorrow is an error, so observed
weather is fetched up to today. Tomorrow's weather only ever comes from the forecasts anyway.

**Charts** (Altair 6; Altair 5 doesn't run on Python 3.14):
- Colours are the first three slots of the validated reference palette, checked with the
  palette validator in light and dark mode (all checks pass; one light-mode colour is below 3:1
  contrast, so every chart has a legend and a "Show data" table).
- Vega-Lite reads a column called `q0.05` as "field `05` inside `q0`", so the band columns are
  renamed (`lo90`, `hi50`, …) before charting.
- Times are shown in each country's local clock for every viewer, by encoding local wall-clock
  times on a UTC scale.
- The "2 days ago" baseline (≈ −120% skill) is left off the skill chart because it squashed
  every other model into a corner. It stays in the data table.

**Testing.** The live job's history logic has unit tests (`tests/test_live.py`). The app was
smoke-tested headlessly with Streamlit's `AppTest` on every page × country × target (25 renders,
no errors), plus a Track record run on 10 days of synthetic history. A full local run of
`energy-forecast live` took 4 minutes on 16 cores.

## Deploying (one-time, by you)

1. **Push** `main` (includes `.github/workflows/live.yml`). If GitHub rejects it with a message
   about the `workflow` scope, your saved git credential needs that permission: create a new
   token at GitHub → Settings → Developer settings → Personal access tokens with the `repo` and
   `workflow` scopes, and use it as the password on the next push.
2. **Run the job once by hand:** repo → Actions → "Live forecast" → "Run workflow". The first
   run downloads all history. On GitHub's servers that took **1 h 35 min** (2026-10-25), mostly
   Open-Meteo answering slowly (72 timeouts, all weather requests). The downloaded data is cached,
   even if a run fails, so later runs only fetch the latest quarter. When it finishes, the repo
   has a `live` branch.
3. **Deploy the app:** share.streamlit.io → "Create app" → pick the repo, branch `main`, main
   file `app/streamlit_app.py` → Deploy. Deployed at
   [eu-electricity-forecast.streamlit.app](https://eu-electricity-forecast.streamlit.app/).
