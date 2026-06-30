import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
sys.path.append("..")
from config import FORECAST_HORIZONS, THRESHOLDS
from database.supabase_client import fetch_latest_observation, fetch_all_observations, fetch_alerts
from features.engineer import classify_risk, engineer_features
from ai_explainer.explainer import explain_forecast, format_predictions
from models.baseline_xgb import predict_xgb

st.set_page_config(
    page_title="ISRO Radiation Monitor",
    page_icon="🛰️",
    layout="wide"
)

RISK_COLORS = {
    "Nominal":  "#2ecc71",
    "Elevated": "#f39c12",
    "Storm":    "#e74c3c",
    "Severe":   "#8e44ad",
    "Unknown":  "#95a5a6"
}

def get_risk_color(risk: str) -> str:
    return RISK_COLORS.get(risk, "#95a5a6")

def load_recent_data(n=200) -> pd.DataFrame:
    df = fetch_all_observations(limit=n)
    return df

# ---- Header ----
st.title("🛰️ ISRO Geostationary Satellite Radiation Monitor")
st.caption("Real-time space weather monitoring and energetic particle radiation forecasting")

# ---- Auto refresh ----
st.markdown("""
<meta http-equiv="refresh" content="300">
""", unsafe_allow_html=True)

# ---- Latest conditions ----
latest = fetch_latest_observation()

if not latest:
    st.error("No data in database. Run `ingestion/noaa_fetcher.py` first.")
    st.stop()

current_risk = classify_risk(latest.get("proton_flux"))

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Bz (IMF)", f"{latest.get('bz', 'N/A')} nT",
            help="Negative = southward = dangerous")
col2.metric("Solar Wind", f"{latest.get('solar_wind_speed', 'N/A')} km/s")
col3.metric("Density", f"{latest.get('density', 'N/A')} p/cm³")
col4.metric("Kp Index", f"{latest.get('kp_index', 'N/A')}")
col5.metric("Proton Flux", f"{latest.get('proton_flux', 'N/A')} pfu")

st.markdown("---")

# ---- Risk badge ----
risk_color = get_risk_color(current_risk)
st.markdown(f"""
<div style='text-align:center; padding: 1rem; background: {risk_color}22;
border: 2px solid {risk_color}; border-radius: 12px; margin-bottom: 1rem;'>
    <h2 style='color: {risk_color}; margin: 0;'>Current Risk: {current_risk}</h2>
    <p style='color: #666; margin: 0;'>as of {latest.get('timestamp', 'N/A')}</p>
</div>
""", unsafe_allow_html=True)

# ---- Forecast ----
st.subheader("📈 Radiation Forecast")

df_recent = load_recent_data(500)
if not df_recent.empty:
    df_eng = engineer_features(df_recent).fillna(0)

    if not df_eng.empty:
        forecast_cols = st.columns(3)
        predictions = {}
        pred_values = {}

        for i, h in enumerate(FORECAST_HORIZONS):
            try:
                latest_row = df_eng.tail(1)
                pred_flux = predict_xgb(latest_row, h)
                pred_risk = classify_risk(pred_flux)
                predictions[f"{h}h"] = {"flux": pred_flux, "risk": pred_risk}
                pred_values[h] = pred_flux
                color = get_risk_color(pred_risk)
                forecast_cols[i].markdown(f"""
<div style='text-align:center; padding:1rem; border:1px solid {color};
border-radius:8px; background:{color}11;'>
    <h3 style='color:{color}; margin:0;'>{h}h Ahead</h3>
    <p style='font-size:1.4rem; font-weight:bold; margin:0.5rem 0;'>{pred_flux:.3f} pfu</p>
    <span style='color:{color}; font-weight:bold;'>{pred_risk}</span>
</div>
""", unsafe_allow_html=True)
            except Exception as e:
                forecast_cols[i].warning(f"{h}h model not trained yet")

        # ---- Ollama explanation ----
        st.subheader("🤖 AI Analyst Explanation")
        if st.button("Generate Explanation"):
            with st.spinner("Asking qwen2.5:7b..."):
                explanation = explain_forecast(latest, predictions)
                st.info(explanation)

st.markdown("---")

# ---- Time series chart ----
st.subheader("📊 Historical Data")

tab1, tab2, tab3 = st.tabs(["Proton Flux", "Bz & Solar Wind", "Kp Index"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_recent["timestamp"], y=df_recent["proton_flux"],
        name="Proton Flux", line=dict(color="#e74c3c")
    ))
    for label, val in THRESHOLDS.items():
        fig.add_hline(y=val, line_dash="dash",
                      annotation_text=label, line_color="gray")
    fig.update_layout(title="Proton Flux History", xaxis_title="Time",
                      yaxis_title="pfu", yaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_recent["timestamp"], y=df_recent["bz"],
        name="Bz (nT)", line=dict(color="#3498db")
    ))
    fig2.add_hline(y=0, line_dash="solid", line_color="red", line_width=1)
    fig2.update_layout(title="Bz Component (negative = southward = dangerous)",
                       xaxis_title="Time", yaxis_title="nT")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_recent["timestamp"], y=df_recent["kp_index"],
        name="Kp Index", line=dict(color="#9b59b6"), fill="tozeroy"
    ))
    fig3.add_hline(y=5, line_dash="dash", annotation_text="Storm threshold",
                   line_color="orange")
    fig3.update_layout(title="Kp Geomagnetic Index",
                       xaxis_title="Time", yaxis_title="Kp")
    st.plotly_chart(fig3, use_container_width=True)

# ---- Alert log ----
st.markdown("---")
st.subheader("🚨 Alert Log")
alerts = fetch_alerts(limit=20)
if alerts.empty:
    st.info("No alerts triggered yet.")
else:
    st.dataframe(alerts, use_container_width=True)

st.caption("Data sources: NOAA SWPC, NASA DONKI, NASA OMNIWeb, ISRO Aditya-L1")
