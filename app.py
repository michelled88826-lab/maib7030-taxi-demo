from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast_engine import (
    load_demo_data,
    train_demo_models,
    get_case_row,
    get_prior_history,
    predict_row,
    reveal_actual,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "taxi_demand_features.csv"
LOOKUP_PATH = ROOT / "data" / "taxi_zone_lookup.csv"

st.set_page_config(
    page_title="NYC Taxi Demand Forecast",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1380px;}
[data-testid="stMetric"] {background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.16); padding: 14px 16px; border-radius: 12px;}
.demo-note {padding: 12px 14px; border-radius: 10px; background: rgba(64,140,220,.09); border: 1px solid rgba(64,140,220,.22);}
.small-muted {font-size: .88rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading and checking the frozen taxi dataset…")
def get_data():
    return load_demo_data(DATA_PATH)


@st.cache_data(show_spinner=False)
def get_zone_names():
    if not LOOKUP_PATH.exists():
        return {}
    lookup = pd.read_csv(LOOKUP_PATH)
    return dict(zip(lookup["LocationID"].astype(int), lookup["Zone"].astype(str)))


@st.cache_resource(show_spinner="Loading frozen January–April models (no training on this computer)…")
def get_models(_df):
    return train_demo_models(_df)


if not DATA_PATH.exists():
    st.error("Dataset not found. Please keep data/taxi_demand_features.csv inside the demo folder.")
    st.stop()

try:
    df = get_data()
    model_bundle = get_models(df)
    zone_names = get_zone_names()
except Exception as exc:
    st.error(f"The demo could not initialize: {exc}")
    st.stop()

st.title("NYC Taxi Demand Forecast")
st.caption("One-hour-ahead zone-level pickup forecasting · Historical validation replay · May 2025")

st.markdown(
    '<div class="demo-note"><b>Evaluation-safe demo:</b> Frozen models were trained only on 8 January–30 April 2025 '
    '(27,110 zone-hour observations). The interface replays May validation hours (7,440 observations). '
    'June and July are excluded from this application.</div>',
    unsafe_allow_html=True,
)

# Sidebar controls
st.sidebar.header("Forecast scenario")
zone_ids = model_bundle["zones"]

def zone_label(z):
    return f"{z} · {zone_names.get(int(z), 'Manhattan taxi zone')}"

with st.sidebar.form("scenario_form"):
    selected_zone = st.selectbox("Taxi zone", zone_ids, format_func=zone_label, index=zone_ids.index(142) if 142 in zone_ids else 0)
    may_dates = [date(2025, 5, 1) + timedelta(days=i) for i in range(31)]
    selected_date = st.selectbox("Target date (May 2025)", may_dates, index=0, format_func=lambda d: d.strftime("%d %b %Y"))
    selected_hour = st.selectbox("Target hour (New York local time)", list(range(24)), index=22, format_func=lambda h: f"{h:02d}:00–{h:02d}:59")
    run_forecast = st.form_submit_button("Run one-hour-ahead forecast", type="primary", use_container_width=True)

if run_forecast:
    target_ts = pd.Timestamp(selected_date) + pd.Timedelta(hours=int(selected_hour))
    try:
        case_row = get_case_row(df, int(selected_zone), target_ts)
        history = get_prior_history(df, int(selected_zone), target_ts, hours=24)
        if history.empty:
            raise ValueError("No preceding hourly history is available for this case.")
        predictions = predict_row(case_row, model_bundle)
        # Do not retain the target-hour actual in session state before reveal.
        feature_row = case_row.drop(labels=["pickup_count"])
        st.session_state["forecast_case"] = {
            "zone": int(selected_zone),
            "target_ts": target_ts,
            "feature_row": feature_row,
            "history": history,
            "predictions": predictions,
        }
        st.session_state["actual_revealed"] = False
    except Exception as exc:
        st.error(str(exc))

case = st.session_state.get("forecast_case")
if case is None:
    st.subheader("Start with a historical forecast")
    st.write("Choose a Manhattan taxi zone, a date in May 2025 and a target hour. The model will use only historical features available before that hour.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Training observations", f"{model_bundle['training_rows']:,}")
    m2.metric("May validation observations", f"{model_bundle['validation_rows']:,}")
    m3.metric("Taxi zones", f"{len(zone_ids)}")
    st.info("Suggested presentation case: Zone 142 · 1 May 2025 · 22:00. This is a documented missed-surge example.")
    st.stop()

feature_row = case["feature_row"]
history = case["history"]
predictions = case["predictions"]
target_ts = case["target_ts"]
zone_id = case["zone"]

st.subheader(f"Forecast result · {zone_label(zone_id)}")
st.caption(f"Target hour: {target_ts.strftime('%A, %d %B %Y · %H:%M')} New York local time")

c1, c2, c3 = st.columns(3)
c1.metric("HGB baseline", f"{predictions['HGB baseline']:.1f}", help="Histogram Gradient Boosting trained on Jan–Apr with the original feature set.")
c2.metric("HGB + transition", f"{predictions['HGB + transition']:.1f}", help="Same model family with two additional recent-demand transition features.")
c3.metric("Previous-hour pickups", f"{int(feature_row['lag_1h']):,}", help="Observed pickups at t−1, available before the target hour.")

st.markdown("#### Recent demand context")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=history["hour"], y=history["pickup_count"], mode="lines+markers",
    name="Observed history (t−24 to t−1)", line=dict(width=2),
))
fig.add_trace(go.Scatter(
    x=[target_ts], y=[predictions["HGB baseline"]], mode="markers",
    name="HGB forecast", marker=dict(size=12, symbol="diamond"),
))
fig.add_trace(go.Scatter(
    x=[target_ts], y=[predictions["HGB + transition"]], mode="markers",
    name="HGB + transition forecast", marker=dict(size=12, symbol="square"),
))
if st.session_state.get("actual_revealed", False):
    actual = reveal_actual(df, zone_id, target_ts)
    fig.add_trace(go.Scatter(
        x=[target_ts], y=[actual], mode="markers+text", name="Actual recorded pickups",
        text=[f"Actual: {actual}"], textposition="top center", marker=dict(size=14, symbol="star"),
    ))
fig.update_layout(
    height=390, margin=dict(l=10, r=10, t=20, b=10),
    xaxis_title="New York local time", yaxis_title="Recorded pickups",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1.1, 1])
with left:
    st.markdown("#### Prediction-time feature context")
    feature_rows = [
        ("Previous hour (lag_1h)", feature_row["lag_1h"]),
        ("Same hour previous day (lag_24h)", feature_row["lag_24h"]),
        ("Same observed hour previous week (lag_168h)", feature_row["lag_168h"]),
        ("Previous 3-hour mean", feature_row["rolling_mean_3h"]),
        ("Previous 24-hour mean", feature_row["rolling_mean_24h"]),
    ]
    st.dataframe(pd.DataFrame(feature_rows, columns=["Feature", "Value"]).round(2), hide_index=True, use_container_width=True)
with right:
    st.markdown("#### May validation reference")
    metric_df = pd.DataFrame(model_bundle["may_metrics"]).T
    st.dataframe(metric_df.rename(columns={"MAE": "MAE ↓", "RMSE": "RMSE ↓", "WAPE": "WAPE ↓"}).round(3), use_container_width=True)
    st.caption("Aggregate May metrics are validation results, not results from the selected case alone.")

st.divider()
if not st.session_state.get("actual_revealed", False):
    st.warning("The target-hour actual is hidden. Reveal it only after discussing the forecast.")
    if st.button("Reveal actual result", type="primary"):
        st.session_state["actual_revealed"] = True
        st.rerun()
else:
    actual = reveal_actual(df, zone_id, target_ts)
    st.markdown("#### Forecast evaluation for this selected hour")
    e1, e2, e3 = st.columns(3)
    e1.metric("Actual recorded pickups", f"{actual:,}")
    e2.metric("HGB absolute error", f"{abs(predictions['HGB baseline'] - actual):.1f}")
    e3.metric("Transition-model absolute error", f"{abs(predictions['HGB + transition'] - actual):.1f}")
    st.caption("The actual is the recorded TLC pickup count for this zone-hour, not total passenger demand or unmet demand.")

if st.button("Reset / choose another scenario"):
    st.session_state.pop("forecast_case", None)
    st.session_state["actual_revealed"] = False
    st.rerun()

with st.expander("Model design and limitations"):
    st.markdown("""
    **Baseline:** Histogram Gradient Boosting Regressor, trained on 27,110 Jan–Apr observations and loaded from a fixed model artifact. Neither model is re-trained on the presentation computer.

    **Proposed modification:** adds `recent_gap_3h = lag_1h − rolling_mean_3h` and
    `relative_gap_3h = recent_gap_3h / (10 + rolling_mean_3h)`.

    **Forecast timing:** all lag and rolling inputs refer to completed hours before the target hour.
    The selected target-hour pickup count is not used to generate the prediction.

    **Interpretation:** the transition model is an evaluated candidate, not a proven robust improvement.
    May MAE decreased only slightly, and earlier rolling-origin checks did not show consistent gains.

    **Deployment limitation:** this is a historical replay using the prepared TLC dataset. A live operational
    system would require timely access to completed-hour pickup counts; the public monthly TLC release is not a live feed.
    """)

st.caption("MAIB7030 Group Project · Technical Track · Historical replay only · June/July are not used by this app")
