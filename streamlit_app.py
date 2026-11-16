from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from prophet import Prophet
import streamlit as st

st.set_page_config(page_title="Shop sales forecast", page_icon="📈", layout="centered")
st.markdown("""
<style>
.block-container {max-width: 960px; padding-top: 2.5rem}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums}
[data-testid="stMetricLabel"] p {font-weight: 600}
</style>""", unsafe_allow_html=True)

@st.cache_data
def read_history():
    return pd.read_csv(Path(__file__).with_name("daily_sales.csv"), parse_dates=["date"]).rename(
        columns={"date": "ds", "sales": "y"}).sort_values("ds")

@st.cache_data(show_spinner=False)
def forecast(history, horizon, holiday_mode):
    model = Prophet(holidays_mode=holiday_mode, interval_width=0.8)
    model.add_country_holidays(country_name="US")
    model.fit(history, seed=42)
    dates = model.make_future_dataframe(periods=horizon, include_history=False)
    np.random.seed(42)
    return model.predict(dates)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

def date_range(series):
    return f"{series.min():%b %d, %Y} – {series.max():%b %d, %Y}"

st.title("Shop sales forecast")
st.write("Forecast daily demand for the course shop, or replay the last 60 days to check how the holiday adjustment performed.")
view = st.radio("Forecast view", ["Future demand", "Holdout backtest"], horizontal=True, label_visibility="collapsed")
data = read_history()
is_backtest = view == "Holdout backtest"

if is_backtest:
    history, actual = data.iloc[:-60], data.iloc[-60:]
    horizon = len(actual)
    st.caption(f"All 60 predictions use sales through {history.ds.max():%b %d, %Y}.")
else:
    history = data
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30)
    st.caption(f"Fictional daily shop sales · Observed {date_range(history.ds)}")

with st.spinner("Fitting the holiday forecast…"):
    tuned = forecast(history, horizon, "multiplicative")
    if is_backtest:
        baseline = forecast(history, horizon, "additive")
        tuned = tuned.merge(actual[["ds", "y"]], on="ds", validate="one_to_one")
        tuned["baseline"] = baseline.yhat.to_numpy()

a, b, c = st.columns(3)
if is_backtest:
    base_mae = (tuned.baseline - tuned.y).abs().mean()
    tuned_mae = (tuned.yhat - tuned.y).abs().mean()
    a.metric("Class model MAE", f"{base_mae:.2f}")
    b.metric("Tuned model MAE", f"{tuned_mae:.2f}")
    c.metric("Error reduction", f"{1-tuned_mae/base_mae:.1%}")
    st.subheader("Predictions against actual sales")
    st.caption(f"{date_range(tuned.ds)} · MAE is measured in sales units per day.")
else:
    peak = tuned.loc[tuned.yhat.idxmax()]
    a.metric("Daily average", f"{tuned.yhat.mean():,.0f} units")
    b.metric("Period total", f"{tuned.yhat.sum():,.0f} units")
    c.metric("Peak date", f"{peak.ds:%b %d}")
    st.subheader("Daily demand")
    st.caption(f"Forecast {date_range(tuned.ds)} · Shading shows the model’s daily 80% prediction interval.")

bands = alt.Chart(tuned).mark_area(color="#42ad9c", opacity=0.18).encode(
    x="ds:T", y=alt.Y("yhat_lower:Q", title="Sales units", scale=alt.Scale(zero=False)), y2="yhat_upper:Q")
if is_backtest:
    series = tuned.rename(columns={"yhat": "Tuned forecast", "y": "Actual", "baseline": "Class forecast"})
    long = series.melt(id_vars="ds", value_vars=["Actual", "Tuned forecast", "Class forecast"], var_name="Series", value_name="Units")
    labels, palette = ["Actual", "Tuned forecast", "Class forecast"], ["#334155", "#008676", "#cb741c"]
else:
    observed = history.tail(28).rename(columns={"y": "Units"}).assign(Series="Observed")
    future = tuned[["ds", "yhat"]].rename(columns={"yhat": "Units"}).assign(Series="Forecast")
    long = pd.concat([observed, future])
    labels, palette = ["Observed", "Forecast"], ["#334155", "#008676"]
lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
    x=alt.X("ds:T", title=None),
    y=alt.Y("Units:Q", title="Sales units", scale=alt.Scale(zero=False)),
    color=alt.Color("Series:N", scale=alt.Scale(domain=labels, range=palette),
                    legend=alt.Legend(orient="bottom", title=None)),
    tooltip=[alt.Tooltip("ds:T", title="Date", format="%b %d, %Y"), "Series:N",
             alt.Tooltip("Units:Q", format=".1f")])
st.altair_chart((bands + lines).properties(height=315), width="stretch")
if is_backtest:
    st.write("The class model subtracts a fixed number of units for each holiday. The tuned model lets that holiday effect grow with the shop’s underlying sales.")
else:
    st.write("The forecast uses the weekly and yearly sales patterns and models U.S. holidays as a percentage of the underlying sales trend.")
table = tuned.rename(columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "80% low", "yhat_upper": "80% high",
                              "y": "Actual", "baseline": "Class forecast"}).round(2)
table["Date"] = table.Date.dt.date
with st.expander("Daily forecast table"):
    st.dataframe(table, hide_index=True, width="stretch")
st.download_button("Download forecast CSV", table.to_csv(index=False), "shop_sales_forecast.csv", "text/csv")
