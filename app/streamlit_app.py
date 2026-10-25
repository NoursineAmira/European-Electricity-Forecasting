"""European electricity forecasts: tomorrow, the live track record, and the backtest.

The app only reads CSV files: live forecasts from the repo's `live` branch (written daily by
GitHub Actions) and backtest scores from `results/`. It never trains a model.
"""

import json
import os
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPO = "NoursineAmira/European-Electricity-Forecasting"
LIVE_URL = f"https://raw.githubusercontent.com/{REPO}/live/"
LIVE_DIR = os.environ.get("LIVE_DIR")  # local folder instead of GitHub, for development

COUNTRIES = {
    "de": ("Germany", "Europe/Berlin"),
    "fr": ("France", "Europe/Paris"),
    "es": ("Spain", "Europe/Madrid"),
    "nl": ("Netherlands", "Europe/Amsterdam"),
}
TARGETS = {"load_mw": ("Electricity demand", "GW", 1000), "price_eur_mwh": ("Day-ahead price", "€/MWh", 1)}
MODELS = {
    "naive_daily": "Same hour, 2 days ago",
    "naive_weekly": "Same hour last week",
    "weekly_mean": "4-week average",
    "mstl": "MSTL (pattern model)",
    "lgbm_noweather": "LightGBM, no weather",
    "lgbm_qr": "LightGBM quantile regression",
    "lgbm": "LightGBM (honest weather)",
    "lgbm_oracle": "LightGBM, perfect weather (oracle)",
}
METHODS = {"empirical": "Past errors (before)", "aci": "ACI thermostat", "cqr": "Conformalized QR"}

# Validated categorical slots (dataviz reference palette, light / dark).
PALETTE = {
    "light": {"series": ["#2a78d6", "#eb6834", "#1baf7a"], "muted": "#52514e"},
    "dark": {"series": ["#3987e5", "#d95926", "#199e70"], "muted": "#c3c2b7"},
}


def colors() -> dict:
    return PALETTE["dark" if st.context.theme.type == "dark" else "light"]


# ---------- data ----------


@st.cache_data(ttl=3600, show_spinner=False)
def read_live(name: str) -> str | None:
    try:
        if LIVE_DIR:
            return (Path(LIVE_DIR) / name).read_text()
        with urlopen(LIVE_URL + name, timeout=10) as resp:
            return resp.read().decode()
    except OSError:
        return None


def live_frame(name: str) -> pd.DataFrame | None:
    text = read_live(name)
    if text is None:
        return None
    return pd.read_csv(StringIO(text), parse_dates=["timestamp"], dtype={"issued": str})


@st.cache_data(show_spinner=False)
def results(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / name)


def as_local_clock(ts: pd.Series, country: str) -> pd.Series:
    """Local wall-clock time, encoded as if UTC so every viewer sees the country's clock."""
    local = pd.to_datetime(ts, utc=True).dt.tz_convert(COUNTRIES[country][1]).dt.tz_localize(None)
    return local.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- charts ----------

TIME_AXIS = alt.Axis(labelExpr="utcFormat(datum.value, '%a %H:%M')", labelAngle=0, title=None, grid=False, tickCount=8)


def forecast_chart(fc: pd.DataFrame, actual: pd.DataFrame, country: str, unit: str, scale: float) -> alt.Chart:
    """Forecast line with 50% and 90% bands, plus actual values, on one axis."""
    c = colors()
    # Vega-Lite reads "q0.05" as a nested field, so give the bands dot-free names.
    bands = {"q0.05": "lo90", "q0.25": "lo50", "q0.75": "hi50", "q0.95": "hi90"}
    wide = fc.rename(columns=bands).assign(time=as_local_clock(fc["timestamp"], country))
    wide["when"] = pd.to_datetime(wide["time"]).dt.strftime("%a %d %b, %H:%M")
    for col in ["forecast", *bands.values()]:
        wide[col] = wide[col] / scale
    lines = pd.concat(
        [
            wide[["time", "forecast"]].rename(columns={"forecast": "value"}).assign(series="Forecast"),
            actual.assign(time=as_local_clock(actual["timestamp"], country), value=actual["actual"] / scale)[
                ["time", "value"]
            ].assign(series="Actual"),
        ]
    )
    x = alt.X("time:T", scale=alt.Scale(type="utc"), axis=TIME_AXIS)
    y_title = unit
    band = lambda lo, hi, opacity: alt.Chart(wide).mark_area(opacity=opacity, color=c["series"][0]).encode(
        x=x, y=alt.Y(f"{lo}:Q", title=y_title, scale=alt.Scale(zero=False)), y2=f"{hi}:Q"
    )
    line = (
        alt.Chart(lines)
        .mark_line(strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["Forecast", "Actual"], range=c["series"][:2]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["time"], empty=False)
    rule = (
        alt.Chart(wide)
        .mark_rule(color=c["muted"])
        .encode(
            x=x,
            opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
            tooltip=[
                alt.Tooltip("when:N", title="Local time"),
                alt.Tooltip("forecast:Q", title=f"Forecast ({unit})", format=",.1f"),
                alt.Tooltip("lo50:Q", title="50% range from", format=",.1f"),
                alt.Tooltip("hi50:Q", title="50% range to", format=",.1f"),
                alt.Tooltip("lo90:Q", title="90% range from", format=",.1f"),
                alt.Tooltip("hi90:Q", title="90% range to", format=",.1f"),
            ],
        )
        .add_params(hover)
    )
    return (band("lo90", "hi90", 0.15) + band("lo50", "hi50", 0.3) + line + rule).properties(height=360)


def model_lines(df: pd.DataFrame, x_field: str, y_field: str, series_field: str, labels: dict, y_title: str,
                reference: float | None = None) -> alt.Chart:
    c = colors()
    data = df[df[series_field].isin(labels)].assign(label=lambda d: d[series_field].map(labels))
    order = list(labels.values())
    hover = alt.selection_point(nearest=True, on="pointerover", fields=[x_field], empty=False)
    base = alt.Chart(data).encode(x=alt.X(f"{x_field}:O", title=None, axis=alt.Axis(labelAngle=0)))
    lines = base.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
        y=alt.Y(f"{y_field}:Q", title=y_title, scale=alt.Scale(zero=False)),
        color=alt.Color("label:N", scale=alt.Scale(domain=order, range=c["series"][: len(order)]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip(f"{x_field}:O", title=x_field.title()), alt.Tooltip("label:N", title="Series"),
                 alt.Tooltip(f"{y_field}:Q", title=y_title, format=",.1f")],
    )
    rule = base.mark_rule(color=c["muted"]).encode(opacity=alt.condition(hover, alt.value(0.4), alt.value(0))).add_params(hover)
    chart = lines + rule
    if reference is not None:
        chart += alt.Chart(pd.DataFrame({"y": [reference]})).mark_rule(color=c["muted"], strokeDash=[4, 4]).encode(y="y:Q")
    return chart.properties(height=320)


def skill_chart(ci: pd.DataFrame) -> alt.Chart:
    c = colors()
    # "2 days ago" (about -120%) would squash the scale; it stays in the data table.
    shown = ci["model"].isin(MODELS) & ~ci["model"].isin(["naive_weekly", "naive_daily"])
    data = ci[shown].assign(label=lambda d: d["model"].map(MODELS))
    order = data.sort_values("skill_%", ascending=False)["label"].tolist()
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=280))
    tooltip = [alt.Tooltip("label:N", title="Model"), alt.Tooltip("skill_%:Q", title="Skill (%)", format=".1f"),
               alt.Tooltip("ci7_lo:Q", title="95% interval from", format=".1f"),
               alt.Tooltip("ci7_hi:Q", title="95% interval to", format=".1f")]
    bars = alt.Chart(data).mark_rule(strokeWidth=2, color=c["series"][0]).encode(
        y=y, x=alt.X("ci7_lo:Q", title="Skill vs “same hour last week” (%)"), x2="ci7_hi:Q", tooltip=tooltip)
    dots = alt.Chart(data).mark_circle(size=90, color=c["series"][0], opacity=1).encode(y=y, x="skill_%:Q", tooltip=tooltip)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=c["muted"], strokeDash=[4, 4]).encode(x="x:Q")
    return (zero + bars + dots).properties(height=40 * len(order) + 40)


def show_data(df: pd.DataFrame, label: str = "Show data") -> None:
    with st.expander(label):
        st.dataframe(df, hide_index=True, width="stretch")


def pickers(key: str, targets=TARGETS) -> tuple[str, str]:
    left, right = st.columns(2)
    country = left.selectbox("Country", list(COUNTRIES), format_func=lambda c: COUNTRIES[c][0], key=f"{key}-c")
    target = right.selectbox("What", list(targets), format_func=lambda t: TARGETS[t][0], key=f"{key}-t")
    return country, target


# ---------- pages ----------


def page_tomorrow() -> None:
    st.title("Tomorrow's forecast")
    history, recent, status_text = live_frame("history.csv"), live_frame("recent.csv"), read_live("status.json")
    if history is None or status_text is None:
        st.info("The first live forecast hasn't been published yet. It appears after the daily job's first run.")
        return
    status = json.loads(status_text)
    country, target = pickers("tomorrow")
    name, unit, scale = TARGETS[target]

    fc = history[(history["issued"] == status["issued"]) & (history["country"] == country) & (history["target"] == target)]
    if fc.empty:
        st.warning(f"{name} for {COUNTRIES[country][0]} isn't published"
                   + (" (Spanish prices can't be shared under their data licence)." if country == "es" else "."))
        return

    today_utc = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    st.caption(f"Issued {status['issued']} (run at {status['run_at_utc']} UTC), using data up to "
               f"{status['data_cut_utc']} UTC only. Times shown in {COUNTRIES[country][0]} local time.")
    if status["issued"] != today_utc:
        st.warning(f"No forecast has been published today yet; this is the latest one, from {status['issued']}.")
    if not status["data_complete"].get(f"{country}/{target}", True):
        st.info("The latest data before the 09:00 UTC cut-off hadn't arrived when this ran, so the model "
                "worked with slightly older information.")

    actual = recent[(recent["country"] == country) & (recent["target"] == target)]
    st.altair_chart(forecast_chart(fc, actual, country, unit, scale), width="stretch")
    st.caption("Line: most likely value. Dark band: 50% range. Light band: 90% range.")
    show_data(fc[["timestamp", "forecast", "q0.05", "q0.25", "q0.75", "q0.95"]])


def page_track_record() -> None:
    st.title("Live track record")
    st.write("Every forecast published since launch, scored against what actually happened. "
             "These were made before the truth existed. Nothing here comes from the backtest.")
    history = live_frame("history.csv")
    if history is None:
        st.info("No live forecasts yet.")
        return
    country, target = pickers("record")
    name, unit, scale = TARGETS[target]
    rows = history[(history["country"] == country) & (history["target"] == target)].dropna(subset=["actual"])
    days = rows["issued"].nunique()
    if days < 7:
        st.info(f"{days} day(s) scored so far. Scores appear once there are 7, so a lucky first day can't look like a result.")
        if days == 0:
            return
    else:
        err = rows["actual"] - rows["forecast"]
        inside = (rows["actual"] >= rows["q0.05"]) & (rows["actual"] <= rows["q0.95"])
        a, b, c = st.columns(3)
        a.metric("Days scored", days)
        if target == "load_mw":
            b.metric("Average error", f"{(err.abs() / rows['actual'].abs()).mean() * 100:.1f}%")
        else:
            b.metric("Average error", f"{err.abs().mean():.1f} €/MWh")
        c.metric("Inside the 90% range", f"{inside.mean() * 100:.0f}%", help="Should be close to 90%.")
    st.altair_chart(forecast_chart(rows, rows[["timestamp", "actual"]], country, unit, scale), width="stretch")
    st.caption("The ranges are recalibrated with the ACI thermostat (load) or conformalized quantile regression (price). "
               "Live, the thermostat only has a few months of history, so expect coverage a little under 90% at first.")
    show_data(rows[["issued", "timestamp", "forecast", "q0.05", "q0.95", "actual"]])


def page_backtest() -> None:
    st.title("Backtest: one full year, replayed day by day")
    st.write("Every day from October 2025 to September 2026, each model forecast the next day using only "
             "information available at 10:00 UTC the day before. Then its forecast was compared with reality.")
    country, target = pickers("backtest")
    name, unit, _ = TARGETS[target]
    unit = "MW" if target == "load_mw" else unit
    pick = lambda df: df[(df["country"] == country) & (df["target"] == target)]

    metrics = pick(results("metrics.csv")).set_index("model")
    ci = pick(results("skill_ci.csv"))
    best = metrics.loc["lgbm"]
    a, b, c = st.columns(3)
    a.metric("LightGBM skill vs “last week”", f"{best['skill_%']:+.0f}%")
    if target == "load_mw":
        b.metric("Average error (MAPE)", f"{best['mape_%']:.1f}%")
    else:
        b.metric("Average error (MAE)", f"{best['mae']:.1f} €/MWh")
    c.metric("Error on public holidays", f"{best['mae_holiday']:,.0f} {unit}",
             delta=f"{best['mae_holiday'] - metrics.loc['naive_weekly', 'mae_holiday']:,.0f} vs last week",
             delta_color="inverse")

    st.subheader("How much better than “same hour last week”?")
    st.caption("Dot: skill over the test year. Line: 95% interval from resampling the year in weekly blocks.")
    st.altair_chart(skill_chart(ci), width="stretch")
    show_data(ci.drop(columns=["country", "target"]).assign(model=lambda d: d["model"].map(MODELS)))

    st.subheader("Average error by month")
    by_month = pick(results("mae_by_month.csv"))
    labels = {m: MODELS[m] for m in ("naive_weekly", "mstl", "lgbm")}
    st.altair_chart(model_lines(by_month, "month", "mae", "model", labels, f"Average error ({unit})"), width="stretch")
    show_data(by_month[by_month["model"].isin(labels)].pivot(index="month", columns="model", values="mae").reset_index())

    st.subheader("Do the 90% ranges hold, month by month?")
    st.caption("Dashed line: the 90% target. Ranges around LightGBM (honest weather).")
    cov = pick(results("coverage_by_month.csv"))
    cov = cov[(cov["gamma"] == 0) | (cov["gamma"] == 0.005)]
    labels = {m: METHODS[m] for m in ("empirical", "aci", "cqr")}
    st.altair_chart(model_lines(cov, "month", "coverage_90_%", "method", labels, "Inside the 90% range (%)", reference=90),
                    width="stretch")
    show_data(cov[cov["method"].isin(labels)].pivot(index="month", columns="method", values="coverage_90_%").reset_index())


def page_method() -> None:
    st.title("How it works")
    st.markdown(f"""
**The question:** every morning, what will electricity demand and prices be for each hour of
tomorrow in Germany, France, Spain and the Netherlands?

**The rules, the same in the backtest and live:**
- The forecast is made at 10:00 UTC and only uses data up to 09:00 UTC.
- Weather comes from forecasts that were published at least two days ahead, not from the weather that actually happened.

**The model:** LightGBM, a gradient-boosted tree model, fed with
- the calendar in local time, public holidays and bridge days;
- the same hour 2, 7 and 14 days earlier, and the latest known values;
- the temperature, wind, sunshine, clouds and humidity forecasts.

**The ranges:** a forecast is a best guess plus a range that should contain the truth 90% of
the time. They're calibrated from the model's own recent errors and adjusted daily (adaptive
conformal inference), so they widen after misses and tighten after hits.

**Data:** [Energy-Charts](https://api.energy-charts.info) (Fraunhofer ISE) and
[Open-Meteo](https://open-meteo.com). Spanish prices are not shown: their licence doesn't allow
sharing them.

**Everything else** (code, tests, and a write-up of every decision): [GitHub](https://github.com/{REPO}).
""")


st.set_page_config(page_title="European electricity forecasts", page_icon="⚡", layout="centered")
st.navigation(
    [
        st.Page(page_tomorrow, title="Tomorrow", url_path="tomorrow", default=True),
        st.Page(page_track_record, title="Track record", url_path="track-record"),
        st.Page(page_backtest, title="Backtest", url_path="backtest"),
        st.Page(page_method, title="How it works", url_path="how-it-works"),
    ]
).run()
