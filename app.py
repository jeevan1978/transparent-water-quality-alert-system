"""
app.py
======
Interactive Streamlit Dashboard for the Transparent Water Quality Alert System.

CRITICAL DISCLAIMER:
--------------------
ALL DATA PRESENTED IN THIS APPLICATION IS 100% SYNTHETIC / SIMULATED.
THIS DASHBOARD DOES NOT CERTIFY OR VERIFY REAL-WORLD DRINKING WATER SAFETY
OR COMPLIANCE WITH STATUTORY PUBLIC HEALTH DIRECTIVES.
"""

from datetime import datetime, timedelta
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.alert_engine import WaterAlertEngine
from src.current_classifier import WaterQualityClassifier
from src.data_simulator import WaterDataSimulator
from src.database import WaterDatabase
from src.forecasting import WaterParameterForecaster
from src.labeling import determine_quality_label, label_dataframe
from src.preprocessing import FEATURE_COLUMNS


# Page Configuration
st.set_page_config(
    page_title="Transparent Water Quality Alert System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mandatory Synthetic Data Notice Banner
st.warning(
    "⚠️ **DEMONSTRATION & PROTOTYPE NOTICE:** "
    "All telemetry data displayed is **100% synthetically simulated**. "
    "This system is built for educational modeling and algorithm testing; "
    "it **does NOT** certify real-world drinking water potability or statutory public health compliance."
)


@st.cache_resource
def get_database():
    return WaterDatabase()


@st.cache_resource
def get_classifier():
    return WaterQualityClassifier()


@st.cache_resource
def get_forecaster_6h():
    return WaterParameterForecaster(horizon_hours=6)


@st.cache_resource
def get_forecaster_24h():
    return WaterParameterForecaster(horizon_hours=24)


@st.cache_resource
def get_alert_engine():
    return WaterAlertEngine()


db = get_database()
classifier = get_classifier()
forecaster_6h = get_forecaster_6h()
forecaster_24h = get_forecaster_24h()
alert_engine = get_alert_engine()


# ------------------------------------------------------------------------------
# Sidebar Navigation & Station Selection
# ------------------------------------------------------------------------------
st.sidebar.title("💧 Water System")
st.sidebar.caption(f"Transparent Alert & Forecasting Platform | **Backend:** {'☁️ Supabase Cloud' if db.mode == 'supabase' else '💾 Local SQLite'}")

stations = ["Station-Alpha", "Station-Beta", "Station-Gamma"]
selected_station = st.sidebar.selectbox("Select Monitoring Station", stations, index=0)

time_window = st.sidebar.radio(
    "Historical Window",
    ["Past 24 Hours", "Past 3 Days", "Past 7 Days", "Past 14 Days"],
    index=1
)

days_map = {
    "Past 24 Hours": 1,
    "Past 3 Days": 3,
    "Past 7 Days": 7,
    "Past 14 Days": 14
}
selected_days = days_map[time_window]

# Simulation Controls in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Telemetry Simulator")
if st.sidebar.button("Generate Fresh Telemetry Batch", width="stretch"):
    with st.spinner("Generating simulated telemetry & forecasts..."):
        sim = WaterDataSimulator(station_id=selected_station)
        fresh_df = sim.generate_timeseries(days=selected_days, interval_minutes=60)
        fresh_df = label_dataframe(fresh_df)
        for _, r in fresh_df.iterrows():
            db.insert_reading(r.to_dict())

        # Also generate and store initial 6h & 24h forecasts for the batch
        f_cat_6h, _, f_params_6h = forecaster_6h.predict_future_category(fresh_df, classifier)
        f_cat_24h, _, f_params_24h = forecaster_24h.predict_future_category(fresh_df, classifier)
        latest_dt = pd.to_datetime(fresh_df.iloc[-1]["timestamp"])
        db.insert_prediction({
            "station_id": selected_station,
            "created_at": latest_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "target_timestamp": (latest_dt + timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
            "horizon_hours": 6,
            "predicted_category": f_cat_6h,
            "forecasted_params": f_params_6h
        })
        db.insert_prediction({
            "station_id": selected_station,
            "created_at": latest_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "target_timestamp": (latest_dt + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"),
            "horizon_hours": 24,
            "predicted_category": f_cat_24h,
            "forecasted_params": f_params_24h
        })
    st.sidebar.success("Generated & stored telemetry + baseline forecasts!")
    st.rerun()

anomaly_choice = st.sidebar.selectbox(
    "Inject Synthetic Anomaly",
    ["runoff (High Turbidity)", "acid_spike (Low pH)", "hypoxia (Low DO)"]
)
if st.sidebar.button("Inject Anomaly Reading Now", width="stretch"):
    sim = WaterDataSimulator(station_id=selected_station)
    atype = anomaly_choice.split()[0]
    now_ts = datetime.now()
    anom_reading = sim.generate_single_reading(now_ts, inject_anomaly=True, anomaly_type=atype)
    cat, reason = determine_quality_label(anom_reading)
    anom_reading["quality_label"] = cat
    anom_reading["label_reason"] = reason
    db.insert_reading(anom_reading)

    # Evaluate alerts and persist with deduplication
    alerts = alert_engine.evaluate_current_reading(selected_station, anom_reading)
    new_alerts_count = 0
    for a in alerts:
        if db.insert_alert(a.to_dict()):
            new_alerts_count += 1

    if new_alerts_count > 0:
        st.sidebar.error(f"Injected anomaly ({atype}) into {selected_station}! ({new_alerts_count} new alert(s) logged)")
    else:
        st.sidebar.warning(f"Injected anomaly ({atype}) into {selected_station}. (Duplicate alert for this event window suppressed)")
    st.rerun()


# ------------------------------------------------------------------------------
# Fetch or Generate Data for Current Station
# ------------------------------------------------------------------------------
readings_limit = selected_days * 24 + 10
df_readings = db.get_recent_readings(selected_station, limit=readings_limit)

if df_readings.empty or len(df_readings) < 5:
    # First-time auto generation so the app is immediately working for beginners
    sim = WaterDataSimulator(station_id=selected_station)
    df_readings = sim.generate_timeseries(days=selected_days, interval_minutes=60)
    df_readings = label_dataframe(df_readings)
    for _, r in df_readings.iterrows():
        db.insert_reading(r.to_dict())


# ------------------------------------------------------------------------------
# Dashboard Header & Real-Time KPIs
# ------------------------------------------------------------------------------
st.title(f"📊 {selected_station} - Quality & Forecasting Console")
latest_reading = df_readings.iloc[-1].to_dict()

# Current Status Prediction
features_input = pd.DataFrame([{col: latest_reading.get(col, 7.0) for col in FEATURE_COLUMNS}])
current_category = classifier.predict(features_input)[0]
_, current_reason = determine_quality_label(latest_reading)

# Color badge for category
category_colors = {
    "Good": "🟢 #28a745",
    "Needs Treatment": "🟡 #ffc107",
    "Poor": "🔴 #dc3545"
}

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Current Status", current_category, help=current_reason)
with kpi2:
    st.metric("Dissolved Oxygen", f"{latest_reading.get('dissolved_oxygen', 0.0):.2f} mg/L")
with kpi3:
    st.metric("Turbidity", f"{latest_reading.get('turbidity', 0.0):.2f} NTU")
with kpi4:
    st.metric("pH", f"{latest_reading.get('ph', 0.0):.2f}")
with kpi5:
    st.metric("Conductivity", f"{latest_reading.get('conductivity', 0.0):.1f} µS/cm")


# ------------------------------------------------------------------------------
# Navigation Tabs
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Telemetry Trends",
    "🔮 Multi-Horizon Forecasting",
    "🚨 Alert Center",
    "📋 Raw Telemetry & Inspector"
])

# ------------------------------------------------------------------------------
# Tab 1: Telemetry Trends
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Historical Parameter Trajectories")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_readings["timestamp"],
        y=df_readings["dissolved_oxygen"],
        name="Dissolved Oxygen (mg/L)",
        mode="lines+markers",
        line=dict(color="#0077b6", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df_readings["timestamp"],
        y=df_readings["turbidity"],
        name="Turbidity (NTU)",
        mode="lines",
        line=dict(color="#d00000", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df_readings["timestamp"],
        y=df_readings["ph"],
        name="pH",
        mode="lines",
        line=dict(color="#38b000", width=2)
    ))

    # Add reference threshold lines
    fig.add_hline(y=4.0, line_dash="dot", line_color="red", annotation_text="Critical DO Minimum (4.0 mg/L)")
    fig.add_hline(y=5.0, line_dash="dot", line_color="orange", annotation_text="Safe Turbidity Limit (5.0 NTU)")

    fig.update_layout(
        title=f"Telemetry Observations - {selected_station} ({time_window})",
        xaxis_title="Time",
        yaxis_title="Measured Value",
        hovermode="x unified",
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig_temp = px.line(
            df_readings, x="timestamp", y="temperature",
            title="Water Temperature (°C) [Diurnal Variation]",
            color_discrete_sequence=["#f77f00"]
        )
        st.plotly_chart(fig_temp, width="stretch")
    with c2:
        fig_cond = px.line(
            df_readings, x="timestamp", y="conductivity",
            title="Electrical Conductivity (µS/cm)",
            color_discrete_sequence=["#7209b7"]
        )
        st.plotly_chart(fig_cond, width="stretch")


# ------------------------------------------------------------------------------
# Tab 2: Multi-Horizon Forecasting (+6h and +24h)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Predictive Forecasting & Future Quality Category")
    st.write(
        "The system estimates future parameter trajectories using multi-output regressors "
        "and projects the expected future water condition at **+6 Hours** and **+24 Hours**."
    )

    cat_6h, reason_6h, params_6h = forecaster_6h.predict_future_category(df_readings, classifier)
    cat_24h, reason_24h, params_24h = forecaster_24h.predict_future_category(df_readings, classifier)

    fc_col1, fc_col2 = st.columns(2)

    with fc_col1:
        st.markdown(f"### ⏱️ Horizon: +6 Hours")
        st.info(f"**Predicted Category:** `{cat_6h}`\n\n*{reason_6h}*")
        st.markdown("**Forecasted Parameters:**")
        p6_df = pd.DataFrame([params_6h]).T.rename(columns={0: "Value (+6h)"})
        st.dataframe(p6_df, width="stretch")

    with fc_col2:
        st.markdown(f"### ⏱️ Horizon: +24 Hours")
        st.info(f"**Predicted Category:** `{cat_24h}`\n\n*{reason_24h}*")
        st.markdown("**Forecasted Parameters:**")
        p24_df = pd.DataFrame([params_24h]).T.rename(columns={0: "Value (+24h)"})
        st.dataframe(p24_df, width="stretch")

    # Forecast Persistence Controls
    st.markdown("---")
    save_fc_col, _ = st.columns([1, 2])
    with save_fc_col:
        if st.button("💾 Persist Current Forecasts to Database", key="save_forecast_btn", width="stretch"):
            raw_latest_ts = latest_reading.get("timestamp") or datetime.now()
            latest_dt = pd.to_datetime(raw_latest_ts)
            target_6h_dt = latest_dt + timedelta(hours=6)
            target_24h_dt = latest_dt + timedelta(hours=24)

            p6_inserted = db.insert_prediction({
                "station_id": selected_station,
                "created_at": latest_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "target_timestamp": target_6h_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_hours": 6,
                "predicted_category": cat_6h,
                "forecasted_params": params_6h
            })
            p24_inserted = db.insert_prediction({
                "station_id": selected_station,
                "created_at": latest_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "target_timestamp": target_24h_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_hours": 24,
                "predicted_category": cat_24h,
                "forecasted_params": params_24h
            })

            if p6_inserted or p24_inserted:
                st.success("✅ Saved +6h and +24h forecasts to database!")
            else:
                st.info("ℹ️ Forecasts for this station and target horizon already exist in database.")
            st.rerun()

    # Stored Forecast History
    st.subheader("📜 Recent Stored Forecasts")
    stored_preds = db.get_recent_predictions(station_id=selected_station, limit=10)
    if stored_preds:
        preds_display = []
        for p in stored_preds:
            preds_display.append({
                "ID": p.get("id"),
                "Created At": p.get("created_at"),
                "Target Time": p.get("target_timestamp"),
                "Horizon": f"+{p.get('horizon_hours')}h",
                "Predicted Category": p.get("predicted_category"),
                "Forecasted Parameters": str(p.get("parsed_params") or p.get("forecasted_params"))
            })
        st.dataframe(pd.DataFrame(preds_display), width="stretch")
    else:
        st.caption("No historical forecasts logged in database yet for this station.")


# ------------------------------------------------------------------------------
# Tab 3: Alert Center
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Operational Alert Feed")

    # Evaluate current real-time alerts
    current_alerts = alert_engine.evaluate_current_reading(selected_station, latest_reading)

    # Evaluate predictive early-warning alerts
    pred_alerts_6h = alert_engine.evaluate_predictive_forecast(
        selected_station, 6, cat_6h, params_6h
    )
    pred_alerts_24h = alert_engine.evaluate_predictive_forecast(
        selected_station, 24, cat_24h, params_24h
    )

    all_active = current_alerts + pred_alerts_6h + pred_alerts_24h

    if not all_active:
        st.success("✅ No active threshold breaches or predictive degradation alerts for this station.")
    else:
        for alert in all_active:
            badge_color = "red" if alert.severity == "CRITICAL" else "orange"
            source_label = "📡 [CURRENT THRESHOLD]" if alert.alert_source == "CURRENT" else "🔮 [PREDICTIVE FORECAST]"
            with st.container():
                st.markdown(
                    f"#### :{badge_color}[[{alert.severity}]] {source_label} {alert.title} ({alert.alert_type})"
                )
                st.write(f"**Description:** {alert.description}")
                st.write(f"👉 **Operator Recommended Action:** *{alert.recommended_action}*")
                st.caption(f"Station: {alert.station_id} | Alert ID: {alert.alert_id} | Source: {alert.alert_source}")
                st.markdown("---")

        # Explicit action to log active alerts without auto-inserting on passive reruns
        col_log_alerts, _ = st.columns([1, 2])
        with col_log_alerts:
            if st.button("💾 Persist Active Alerts to Database", key="save_active_alerts_btn", width="stretch"):
                logged_count = 0
                for a in all_active:
                    if db.insert_alert(a.to_dict()):
                        logged_count += 1
                if logged_count > 0:
                    st.success(f"✅ Stored {logged_count} new alert(s) in database.")
                else:
                    st.info("ℹ️ All active alerts already exist in database (duplicates suppressed).")
                st.rerun()

    # Historical alerts from database
    st.subheader("Historical Logged Alerts")
    source_filter = st.radio(
        "Filter Logged Alerts by Source",
        ["All", "CURRENT", "PREDICTIVE"],
        horizontal=True,
        key="alert_source_filter"
    )
    selected_source = None if source_filter == "All" else source_filter
    db_alerts = db.get_active_alerts(station_id=selected_station, alert_source=selected_source, limit=15)
    if db_alerts:
        alerts_display = []
        for al in db_alerts:
            alerts_display.append({
                "Alert ID": al.get("alert_id"),
                "Source": al.get("alert_source", "CURRENT"),
                "Severity": al.get("severity"),
                "Type": al.get("alert_type"),
                "Title": al.get("title"),
                "Timestamp": al.get("timestamp"),
                "Action": al.get("recommended_action")
            })
        st.dataframe(pd.DataFrame(alerts_display), width="stretch")
    else:
        st.caption(f"No historical {source_filter if source_filter != 'All' else ''} alerts logged in database yet.")


# ------------------------------------------------------------------------------
# Tab 4: Raw Telemetry & Inspector
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Raw Telemetry Observations")
    st.dataframe(df_readings.sort_values("timestamp", ascending=False), width="stretch")

    csv_data = df_readings.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Current Dataset as CSV",
        data=csv_data,
        file_name=f"{selected_station}_telemetry.csv",
        mime="text/csv"
    )
