"""MAIB7030 | Presentation Edition (v2.0), Streamlit.

UI-only upgrade: uses the EXISTING validated forecast_engine.py, CSV, and frozen
Jan-Apr HGB model files. Does not retrain, edit data, or touch June/July.

Two user-selected historical May case paths demonstrate a missed demand surge and
an ordinary lower-error forecast. All target-hour labels stay hidden until reveal.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast_engine import (
    get_case_row,
    get_prior_history,
    load_demo_data,
    predict_row,
    reveal_actual,
    train_demo_models,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "taxi_demand_features.csv"
LOOKUP = ROOT / "data" / "taxi_zone_lookup.csv"

st.set_page_config(
    page_title="NYC Taxi | Forecast Reliability Lab",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Restrained, projector-readable visual hierarchy. All actual figures come from the models.
st.markdown(
    """
    <style>
      .block-container {max-width:1440px; padding-top:1.4rem; padding-bottom:3rem;}
      [data-testid="stMetric"] {background:rgba(107,143,176,.09); border:1px solid rgba(137,171,200,.18);
        padding:1.0rem 1.15rem; border-radius:14px;}
      [data-testid="stMetricLabel"] {font-size:.9rem;}
      [data-testid="stMetricValue"] {font-size:2.05rem;}
      div[data-testid="stSidebar"] {border-right:1px solid rgba(150,171,191,.15);}
      .eyebrow {font-weight:700; font-size:.77rem; text-transform:uppercase; letter-spacing:.16em;
        color:#55D6C2; margin:.2rem 0 .65rem 0;}
      .helper {color:#A8BBCF;font-size:.94rem;line-height:1.45;}
      .panel {background:rgba(60,101,147,.11);border:1px solid rgba(130,168,205,.24);
        padding:15px 20px;border-radius:12px;line-height:1.5;}
      .notice {background:rgba(212,157,73,.11);border:1px solid rgba(212,157,73,.35);
        border-radius:10px;padding:14px 17px;}
      .tiny {font-size:.78rem;color:#9BB3CA;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner="Checking the Jan–May historical data…")
def get_data() -> pd.DataFrame:
    return load_demo_data(DATA)

@st.cache_data(show_spinner=False)
def get_zone_names() -> dict[int, str]:
    if not LOOKUP.exists():
        return {}
    names = pd.read_csv(LOOKUP)
    return dict(zip(names["LocationID"].astype(int), names["Zone"].astype(str)))

@st.cache_resource(show_spinner="Loading verified Jan–Apr frozen models…")
def get_bundle():
    # In v1.2, train_demo_models is a backward-compatible name for LOADING
    # signed frozen models; no fitting takes place in the presentation app.
    return train_demo_models(get_data())

try:
    df = get_data()
    bundle = get_bundle()
    zone_names = get_zone_names()
except Exception as error:
    st.error(f"Cannot initialize the verified demo: {error}")
    st.info("Check models/manifest.json, both frozen .joblib files and the Jan–May CSV. Do not alter the test data or validation thresholds.")
    st.stop()

zone_ids = sorted(int(x) for x in bundle["zones"])

def zone_text(zone: int) -> str:
    return f"{zone} · {zone_names.get(int(zone), 'Manhattan taxi zone')}"

EXAMPLES = {
    "Missed demand surge · Zone 142": (142, date(2025, 5, 1), 22),
    "Lower-error comparison · Zone 161": (161, date(2025, 5, 15), 19),
}

with st.sidebar:
    st.markdown('<div class="eyebrow">Live case selector</div>', unsafe_allow_html=True)
    st.markdown("### Replay a historical hour")
    st.caption("These are development-validation cases from May 2025, not independent final-test observations.")
    # Outside the form so selecting Custom immediately exposes date/zone widgets.
    mode = st.radio(
        "Scenario",
        options=[*EXAMPLES.keys(), "Choose a custom May hour"],
        index=0,
        help="The featured cases are illustrative; all ten zones and May hours are available through Custom.",
    )
    with st.form("forecast_controls"):
        if mode == "Choose a custom May hour":
            zone = st.selectbox("Taxi zone", zone_ids, format_func=zone_text,
                                index=zone_ids.index(142) if 142 in zone_ids else 0)
            dates = [date(2025, 5, 1) + timedelta(days=d) for d in range(31)]
            day = st.selectbox("Date", dates, format_func=lambda x: x.strftime("%d %b %Y"))
            hour = st.selectbox("Hour (New York time)", list(range(24)), index=22,
                                format_func=lambda h: f"{h:02d}:00–{h:02d}:59")
        else:
            zone, day, hour = EXAMPLES[mode]
            st.markdown(f"**Taxi zone**  {zone_text(zone)}")
            st.markdown(f"**Target hour**  {day.strftime('%d %b %Y')} · {hour:02d}:00")
        go_forecast = st.form_submit_button("01  Run live model inference", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**Model protocol**")
    st.caption("Training: Jan 8–Apr 30 · 27,110 zone-hours")
    st.caption("Historical replay: May · 7,440 zone-hours")
    st.caption("June/July remain excluded from this app")
    st.caption("Frozen HGB baseline and transition candidate")

if go_forecast:
    target = pd.Timestamp(day) + pd.Timedelta(hours=int(hour))
    try:
        row = get_case_row(df, int(zone), target)
        history = get_prior_history(df, int(zone), target, hours=24)
        if history.empty or not bool((history["hour"] < target).all()):
            raise ValueError("Valid observed history before the target hour could not be retrieved.")
        preds = predict_row(row, bundle)
        # Never retain the target actual in the page state prior to reveal.
        st.session_state["case"] = {
            "zone": int(zone),
            "target": target,
            "features": row.drop(labels=["pickup_count"]),
            "history": history,
            "predictions": preds,
            "case_type": mode,
        }
        st.session_state["revealed"] = False
    except Exception as exc:
        st.error(f"Forecast not available: {exc}")

st.markdown('<div class="eyebrow">MAIB7030 · Machine learning group project</div>', unsafe_allow_html=True)
st.title("Taxi Pickup Forecast Reliability")
st.caption("Real model inference  ·  Historical NYC validation replay  ·  Predict first, reveal actual second")
st.markdown(
    '<div class="panel"><b>Research question.</b> How reliable are next-hour recorded taxi pickup forecasts, '
    'especially when recent demand changes abruptly? The comparison shows what a small feature improvement '
    'can—and cannot—resolve.</div>',
    unsafe_allow_html=True,
)
st.write("")

case = st.session_state.get("case")
if case is None:
    st.subheader("01  Select a scenario and generate a forecast")
    a,b,c=st.columns(3)
    a.metric("Training observations", f"{bundle['training_rows']:,}")
    b.metric("May validation observations", f"{bundle['validation_rows']:,}")
    c.metric("Manhattan taxi zones", len(zone_ids))
    st.info("Suggested presentation flow: missed surge (Zone 142) → reveal actual → optionally compare lower-error Zone 161.")
    st.stop()

pred=case["predictions"]
zone=case["zone"]
target=case["target"]
features=case["features"]
history=case["history"]
base=float(pred["HGB baseline"])
improved=float(pred["HGB + transition"])
revealed=bool(st.session_state.get("revealed", False))

st.markdown('<div class="eyebrow">01 / 02 · Forecast before revealing actual</div>', unsafe_allow_html=True)
st.subheader(f"{zone_text(zone)}  |  {target.strftime('%d %b %Y · %H:%M')}")
st.caption("Model inference uses only target-time calendar information and observed pickups from completed earlier hours.")
a,b,c=st.columns(3)
a.metric("HGB baseline · forecast", f"{base:,.1f}", help="Recorded pickups per zone-hour predicted using the original feature set.")
b.metric("HGB + transition · forecast", f"{improved:,.1f}", help="Same architecture with two additional historical demand-change features.")
c.metric("Last observed hour", f"{int(features['lag_1h']):,}", help="Actual recorded pickups at t−1; target-hour actual is not used in either forecast.")

# Plotly rendering; true target point is absent until the reveal button is clicked.
actual = reveal_actual(df, zone, target) if revealed else None
fig=go.Figure()
fig.add_trace(go.Scatter(x=history["hour"],y=history["pickup_count"],
    mode="lines+markers", name="History through t−1", line=dict(color="#4FD7D0",width=3),
    marker=dict(size=5), hovertemplate="%{x}<br>%{y:.0f} pickups<extra>Observed</extra>"))
fig.add_trace(go.Scatter(x=[target],y=[base], mode="markers",name="HGB baseline",
    marker=dict(symbol="diamond",size=15,color="#8DAAFF"),hovertemplate="HGB %{y:.1f}<extra></extra>"))
fig.add_trace(go.Scatter(x=[target],y=[improved], mode="markers",name="HGB + transition",
    marker=dict(symbol="square",size=15,color="#EEB666"),hovertemplate="HGB + transition %{y:.1f}<extra></extra>"))
if revealed:
    fig.add_trace(go.Scatter(x=[target],y=[actual],mode="markers+text",name="Actual (revealed)",
        marker=dict(symbol="star",size=19,color="#F07D86"),
        text=[f"Actual {actual:,}"],textposition="top center",
        textfont=dict(size=14,color="#F6B1B7"),hovertemplate="Actual %{y:.0f}<extra></extra>"))
# Keep chart coordinates in naive NYC local clock time, consistent with the CSV.
# Using target.timestamp()*1000 converts a naive datetime through the SERVER timezone,
# shifting the vertical line on machines outside New York (e.g., 22:00 → 06:00 next day).
forecast_x = target.strftime("%Y-%m-%d %H:%M:%S")
fig.add_shape(type="line", xref="x", yref="paper", x0=forecast_x, x1=forecast_x,
    y0=0, y1=1, line=dict(color="#768BA5", width=1, dash="dot"))
# Explicit local-time range prevents the plot from showing hours after the forecast.
plot_start = pd.Timestamp(history["hour"].min()).strftime("%Y-%m-%d %H:%M:%S")
plot_end = (target + pd.Timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S")
fig.update_layout(height=390,margin=dict(l=12,r=15,t=18,b=10),
    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#D6E4F2",size=12),
    xaxis=dict(title="New York local time",range=[plot_start, plot_end],gridcolor="rgba(140,180,215,.08)",showline=False),
    yaxis=dict(title="Recorded pickups per zone-hour",gridcolor="rgba(140,180,215,.15)",rangemode="tozero"),
    legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0),
    hovermode="x unified")
st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

if not revealed:
    st.markdown('<div class="notice"><b>Target-hour actual is still hidden.</b> These are predictions made from earlier history. Now reveal the recorded outcome to inspect the error.</div>', unsafe_allow_html=True)
    if st.button("02  Reveal actual and evaluate forecast",type="primary",use_container_width=True):
        st.session_state["revealed"]=True
        st.rerun()
else:
    st.markdown('<div class="eyebrow">02 / 02 · Evaluation and decision relevance</div>',unsafe_allow_html=True)
    base_error=abs(base-actual)
    imp_error=abs(improved-actual)
    a,b,c=st.columns(3)
    a.metric("Actual recorded pickups",f"{actual:,}")
    b.metric("HGB absolute error",f"{base_error:.1f}")
    c.metric("Transition absolute error",f"{imp_error:.1f}")
    ref=base
    if actual > ref:
        risk=("Underprediction indicates potential under-allocation risk if this forecast directly influenced "
              "vehicle positioning. It does not establish how many vehicles were needed or how long passengers waited.")
    elif actual < ref:
        risk=("Overprediction indicates potential over-allocation risk if this forecast directly influenced "
              "vehicle positioning. We cannot quantify vehicle costs without actual supply data.")
    else:
        risk="For this case the baseline matches recorded pickups, but other hours can still produce material errors."
    st.markdown(f'<div class="panel"><b>Business interpretation.</b> {risk}</div>',unsafe_allow_html=True)
    if base_error>0:
        change=base_error-imp_error
        st.caption(f"In this **single illustrative hour**, the transition model changes absolute error by {change:+.1f} pickups (positive means lower error). This is not the May-average improvement.")
    st.caption("Recorded pickups do not measure unmet passenger demand. No vehicle savings, waiting-time benefits, or causal explanations are claimed.")

with st.expander("Evidence and forecasting assumptions",expanded=False):
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**Historical inputs available before the target hour**")
        details=[("Pickup count at t−1", features["lag_1h"]),
                 ("Pickup count at t−24 observed-hour lag", features["lag_24h"]),
                 ("Pickup count at t−168 observed-hour lag", features["lag_168h"]),
                 ("Prior three-hour mean", features["rolling_mean_3h"]),
                 ("Prior 24-hour mean", features["rolling_mean_24h"])]
        st.dataframe(pd.DataFrame(details,columns=["Feature","Value"]).round(2),
                     hide_index=True,use_container_width=True)
    with c2:
        st.markdown("**Aggregate May development-validation results**")
        measures=pd.DataFrame(bundle["may_metrics"]).T
        measures.index.name="Model"
        st.dataframe(measures.round(3),use_container_width=True)
        st.caption("These metrics were measured over the same 7,440 May observations. They are distinct from this one case.")
    st.caption("Recent-transition features: lag_1h − rolling_mean_3h, and that difference divided by (10 + rolling_mean_3h). Two models were trained on Jan–Apr only; frozen models are loaded without re-training. June/July are excluded from this demo. A live deployment would require timely completed-hour pickup records.")

st.divider()
if st.button("Reset → choose a new historical hour",use_container_width=False):
    st.session_state.pop("case",None)
    st.session_state["revealed"]=False
    st.rerun()
st.caption("MAIB7030 · Technical Track · Historical replay, not a live NYC TLC service · Research prototype only")
