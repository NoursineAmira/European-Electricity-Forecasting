# 05 · Calibrated intervals and significance tests

## What this step does, in one sentence

It makes the "90% sure" ranges actually right 90% of the time, and checks with statistics that
LightGBM's lead over the baselines is real, not luck.

## Why it matters

Two weak spots remain after step 3:

1. **The ranges are overconfident.** A 90% range should contain the truth 90% of the time. Ours
   manage 84–87%. A grid operator who trusts them will be caught out more often than promised.
   For anyone buying or selling electricity, a range you can't trust is worse than none.
2. **One test year is one sample.** "+52% skill" sounds convincing, but how much would it move
   with a different year? Without an error bar or a test, a sceptical reader can't tell.

Both are what separates "I trained a model" from "I can tell you how much to trust it". That's
what data science interviews probe.

## Spec

### Part A: interval methods

All methods wrap the **honest `lgbm`** from step 3: same exam, same test year, same
"known at 10:00 the day before" rule. Each produces the same 7 quantiles (5%…95%), so they're
scored identically.

| Method | Idea in plain words |
|---|---|
| `empirical` | Today's method: for each hour of day, the spread of the last 8 weeks' errors. |
| `aci` | **Adaptive conformal inference.** Same as `empirical`, plus a thermostat: when the ranges recently missed too often, widen them; when they caught everything, tighten them. |
| `qr` | **Quantile regression.** Train LightGBM to predict the 5% value, the 10% value, and so on directly (one model per quantile). The model can learn that uncertainty is bigger in winter, or on windy days. |
| `cqr` | **Conformalized quantile regression.** `qr`, then corrected by the same thermostat as `aci`. The model's own sense of uncertainty, plus a safety net. |

**How the thermostat (ACI) works.** For each quantile, say the 95% upper bound, we track one
number: the level of past errors we use. Every day, for each hour:

```
if the truth landed above the bound:  raise the level a little   (the bound was too low)
else:                                 lower it a tiny bit
```

The step sizes are set so that, in the long run, the truth lands above the 95% bound exactly 5%
of the time. This holds *whatever the data does*: no assumption that tomorrow looks like the
past. That's the property we're after.

- Step size `γ = 0.005`, the value from the original paper (Gibbs & Candès, 2021). Not tuned,
  see review point 3.
- One thermostat per quantile and per hour of day (7 × 24 = 168 small trackers), since evening
  hours behave differently from night hours.
- Feedback arrives with the same 2-day delay as everything else: the thermostat for day T only
  knows whether the ranges covered up to day T−2.

### Part B: interval scoring

On top of step 2's coverage, width and pinball loss:

| Metric | Plain meaning |
|---|---|
| Coverage at 50% and 90% | How often the truth lands inside. Should be ≈ 50% and ≈ 90%. |
| **Winkler score** (90%) | Width of the range, plus a penalty whenever the truth falls outside (bigger the further out). One number rewarding "narrow *and* right". Lower is better. |
| Coverage by month | A method can hit 90% over the year by being 99% in summer and 75% in winter. This shows whether it's right *in every season*. |

### Part C: is LightGBM's lead real?

**Diebold–Mariano test.** The standard test for "is forecaster A really better than B, or did
B just have bad luck?"

1. For each day, compute both models' average error over the 24 hours. Daily, because the 24
   hours of one forecast share the same mistakes and aren't independent.
2. Take the difference A − B for each of the ~365 days.
3. Test whether its average is really different from zero, allowing for neighbouring days being
   alike (Newey–West variance, 7-day lag), with the Harvey–Leybourne–Newbold small-sample fix.

Comparisons, per country × target:
- `lgbm` vs every baseline (`naive_weekly`, `weekly_mean`, `mstl`): is the ML worth it?
- `lgbm` vs `lgbm_noweather`: is weather worth it?
- `lgbm` vs `lgbm_oracle`: is the cost of weather uncertainty real or noise?

**Error bars on skill.** A 95% confidence interval for each headline skill number, from a
**block bootstrap**: resample the test year in 7-day blocks (so the weekly pattern stays
intact) 2,000 times and see how much the skill moves. The README then shows "+52% [48, 55]"
instead of "+52%".

### Code shape

```
src/energy_forecast/
  intervals.py       empirical (moved from backtest.py), aci, cqr; plus the Winkler score
  significance.py    diebold_mariano(), block_bootstrap_skill()
  models/lgbm.py     + `lgbm_qr`: one LightGBM per quantile (objective="quantile")
  backtest.py        models may return quantiles directly (qr); otherwise they're added afterwards
  experiment.py      + intervals() and significance() steps
```

- `energy-forecast backtest --models lgbm_qr`, then `energy-forecast intervals` → `results/intervals.csv`, `results/coverage_by_month.csv`
- `energy-forecast significance` → `results/significance.csv`, `results/skill_ci.csv`
- New dependency: none. `scipy` is already installed with statsforecast; it's added to
  `pyproject.toml` explicitly since we now import it directly.

### Out of scope

Chronos (step 4). Intervals for the baselines and other LightGBM versions: they keep their
`empirical` ranges, since the point here is choosing a method, not ranking every model again.

## Review

1. **Delayed feedback.** ACI's guarantee is proved for instant feedback; ours is 2 days late.
   Delayed versions keep the long-run guarantee with a slightly slower reaction (Zaffran et al.,
   2022). The 2-day delay is enforced in code, and a test checks a miss on day T can't move the
   bounds before day T+2.
2. **What "guaranteed" means here.** Classic conformal prediction assumes the data is
   "exchangeable" (order doesn't matter). Time series aren't, so plain conformal methods lose
   their guarantee, which is one reason `empirical` under-covers. ACI only promises the *yearly
   average*, not every month. So coverage by month is reported next to it, and if ACI is
   right on average but wrong in winter, that's stated.
3. **γ is not tuned.** Tuning it on the test year would be peeking. Tuning it on the validation
   period would need a second full backtest. *Decided:* use the paper's default, and show a
   sensitivity table (γ = 0.001, 0.005, 0.02) clearly labelled "not used to choose". If the
   results barely move with γ, that's reassuring. If they move a lot, it's a finding to report.
4. **Thin tails.** 8 weeks × 1 value per day = 56 errors per hour. The 5% quantile of 56 numbers
   rests on 2–3 points, so it's noisy and tends to sit too close to the centre. That's the
   second reason `empirical` under-covers. ACI compensates over time; `qr` sidesteps it by
   learning from all training data.
5. **ACI can push a level past 0% or 100%.** When it does, there's no matching past error.
   *Decided:* clip to the most extreme error in the window and report how often it happened.
   An occasional clip is fine. A lot of clipping would mean γ is too large.
6. **Crossing quantiles.** Separately trained quantile models can disagree, e.g. a predicted 90%
   value below the 75% one. *Decided:* sort the 7 values per hour (the standard fix), and
   report how often it was needed.
7. **Many tests at once.** 4 countries × 2 targets × 5 comparisons = 40 tests. At a 5% level,
   about 2 would come out "significant" by chance. *Decided:* Holm correction across all 40,
   and report both raw and corrected p-values.
8. **Bootstrap blocks.** Too-short blocks break the weekly pattern and make the error bars look
   too narrow. 7-day blocks keep it. We also check with 14-day blocks; if the interval barely
   changes, 7 is enough.
9. **Cost.** `lgbm_qr` trains 7 models where `lgbm` trains 1: about 61 retrains × 7 quantiles ×
   8 series ≈ 3,400 fits. With one thread per process, the step 3 run did ~1,460 fits in
   9 minutes, so expect ~20–25 minutes. Everything else (ACI, tests, bootstrap) works on stored
   forecasts and takes seconds.
10. **Comparing `cqr` fairly.** `cqr` uses `qr` as its starting point, so if `qr` is already
    well calibrated, `cqr` can only add noise. Reported as is. Whichever method has the best
    Winkler score *with* coverage within ±2 points of target becomes the default.

## Results (test year Oct 2025 → Sep 2026)

Full tables: [`results/intervals.csv`](../results/intervals.csv),
[`coverage_by_month.csv`](../results/coverage_by_month.csv),
[`significance.csv`](../results/significance.csv), [`skill_ci.csv`](../results/skill_ci.csv).
The quantile-regression backtest (~3,400 fits) took 28 minutes.

### Intervals: 90% coverage (target 90) and Winkler score (lower is better)

| | DE load | FR load | ES load | NL load | DE price | FR price | ES price | NL price |
|---|---|---|---|---|---|---|---|---|
| `empirical` | 85.3 · 7,775 | 87.0 · **6,824** | 85.4 · **4,061** | 84.2 · 2,341 | 84.9 · 138 | 84.8 · 146 | 85.2 · 107 | 84.3 · 132 |
| `qr` | 66.2 · 10,848 | 72.6 · 8,115 | 68.5 · 5,361 | 65.7 · 3,098 | 67.9 · 158 | 67.7 · 168 | 73.2 · 114 | 74.5 · 136 |
| `aci` | **88.2** · 7,790 | **89.7** · 6,903 | **88.1** · 4,152 | **88.0** · **2,288** | 88.0 · 142 | 87.4 · 148 | 88.0 · 107 | 87.6 · 136 |
| `cqr` | 88.0 · 8,639 | 88.7 · 7,052 | 88.9 · 4,198 | 87.4 · 2,373 | **88.1** · **131** | **87.8** · **138** | **87.8** · **97** | **88.2** · **127** |

(Winkler in MW for load, €/MWh for price.)

### What the numbers say

1. **Quantile regression on its own is badly overconfident.** Its "90%" ranges catch the truth
   only 66–75% of the time, and its "50%" ranges 25–34%. Each quantile model learns from its
   training data, where it has already seen the answers, so it underestimates how wrong it'll be
   on new days. The 7 models also disagree constantly: the quantiles crossed and had to be
   sorted in 66–75% of hours for load and 40–43% for price (7 of 8 series; the DE load count
   was cut from the run log). This is the textbook reason conformal correction exists.
2. **Conformal correction rescues it.** `cqr` lifts the same quantile models from ~70% to ~88%
   coverage, and for **price it gives the best Winkler score and pinball loss in every
   country**. Its ranges are narrow on calm days and wide on volatile ones, which a fixed
   error spread can't do.
3. **For load, the thermostat on the plain forecast (`aci`) is the best choice.** It lifts
   coverage from 84–87% to 88–90% for almost no extra width. Load uncertainty is fairly
   steady, so modelling it separately (`cqr`) only adds noise.
4. **Still slightly under 90%.** With the paper's step size (γ = 0.005), the thermostat moves
   slowly: one update per day, with feedback 2 days late. Over one year it gets from ~85% to
   ~88%, not all the way. The sensitivity check (*not used to choose*) shows the trade-off:

   | γ | Coverage (load / price) | Winkler (load / price) |
   |---|---|---|
   | 0.001 | 86.5 / 85.7 | 5,237 / 131 |
   | **0.005** (used) | 88.5 / 87.8 | 5,284 / 133 |
   | 0.02 | 89.8 / 89.3 | 5,446 / 139 |

   (ACI, averaged over countries.) A bigger step reaches 90% but widens the ranges. The paper's
   long-run guarantee also only kicks in over long horizons: with γ = 0.005 and one year of
   data, its worst-case bound is loose. Worth knowing before trusting "guaranteed coverage".
5. **Coverage by month: the season changes are the weak spot.** All methods dip in March
   (78–84%), September (78–85%) and November (83–85%), and overshoot in May (91.7–93.6%). The ranges
   are built from the *previous* 8 weeks, which lag behind when the season turns. March also
   brings the clock change. `aci` is closer to 90% than `empirical` in 10 of 12 months (not in
   February and May, where it overshoots further) but doesn't remove the pattern. A next step would be season-aware scores (e.g. temperature-based).

**Decision (rule from the spec: best Winkler among methods within ±2 points of 90%):**
- **Load → `aci`.** Within the band in all four countries (88.0–89.7%).
- **Price → `cqr`.** Best Winkler everywhere. Within the band in DE and NL; FR and ES land at
  87.8%, 0.2 points short, as does every other method there. Reported as is, not rounded up.

### Is LightGBM's lead real? (Diebold–Mariano, Holm-corrected)

| `lgbm` vs | Series where `lgbm` is significantly better (p_holm < 0.05) |
|---|---|
| `naive_weekly`, `weekly_mean`, `mstl` | **8 of 8**: all p_holm < 0.001 |
| `lgbm_noweather` | **8 of 8**: weather is worth it everywhere |
| `lgbm_oracle` (perfect weather) | the oracle is significantly better in **3 of 8** (DE price, ES load, NL load); in the other 5 the gap is too small to tell from noise |

So the "cost of weather uncertainty" from step 3 is real but small where it shows up at all
(3–6% of MAE: DE price 5.7%, ES load 3.4%, NL load 3.0%), and undetectable in the other five
series (0.4–3.5%).

### Error bars on skill (95%, 7-day block bootstrap)

| Skill vs "last week" | DE | FR | ES | NL |
|---|---|---|---|---|
| `lgbm`, load | 52% [45, 57] | 68% [62, 73] | 50% [44, 54] | 49% [45, 54] |
| `lgbm`, price | 41% [35, 45] | 34% [28, 40] | 36% [29, 43] | 42% [37, 47] |
| `mstl`, load | 20% [11, 29] | 49% [41, 57] | 15% [2, 25] | 11% [0, 20] |
| `mstl`, price | 6% [−1, 14] | 20% [12, 27] | 19% [10, 30] | 4% [−3, 11] |

- LightGBM's intervals never overlap MSTL's for load, and sit far above zero everywhere.
- MSTL's price skill in DE and NL includes zero: its step 2 "win" there may have been luck.
- 14-day blocks give almost the same intervals (within ~2.5 points), so 7-day blocks are enough
  (review point 8).
