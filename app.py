import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import os, json


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("swasthya_ai_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------
# TRAIN PROPHET FOR ER
# ---------------------------------------------------------
@st.cache_data
def train_and_forecast(df, horizon_days: int = 14):
    df_prophet = df.rename(columns={"date": "ds", "er_visits": "y"})
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    m = Prophet()
    m.add_regressor("aqi")
    m.add_regressor("temp_c")
    m.add_regressor("festival")

    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=horizon_days)
    future["aqi"] = df_prophet["aqi"].iloc[-1]
    future["temp_c"] = df_prophet["temp_c"].iloc[-1]
    future["festival"] = 0

    forecast = m.predict(future)
    return m, forecast


# ---------------------------------------------------------
# STAFFING RECOMMENDATION ENGINE
# ---------------------------------------------------------
def generate_staffing_recommendations(forecast, df, capacity_multiplier):
    forecast_14 = forecast[["ds", "yhat"]].tail(14).copy()
    forecast_14["ds"] = pd.to_datetime(forecast_14["ds"])

    latest = df.iloc[-1]
    er_staff = latest["er_staff_capacity"]
    er_capacity = er_staff * capacity_multiplier

    recs = []
    for _, row in forecast_14.iterrows():
        predicted = row["yhat"]
        date = row["ds"].date()
        overload = (predicted - er_capacity) / er_capacity * 100

        if overload < -5:
            action = "No action needed"
        elif overload < 5:
            action = "Normal day: Monitor staffing"
        elif overload < 15:
            action = "Add 1 extra nurse to ER shift"
        elif overload < 25:
            action = "Add 1 nurse + 1 medical officer"
        else:
            action = "CRITICAL: Activate ER surge staffing protocol"

        recs.append({
            "date": date,
            "predicted_er_visits": round(predicted),
            "er_capacity_patients_per_day": er_capacity,
            "overload_percent": round(overload, 1),
            "recommended_action": action
        })

    return pd.DataFrame(recs)


# ---------------------------------------------------------
# INVENTORY ENGINE
# ---------------------------------------------------------
def generate_inventory_recommendations(df):
    last7 = df.tail(7)

    oxygen_daily_usage = last7["oxygen_used"].mean()
    n95_daily_usage = last7["n95_used"].mean()
    para_daily_usage = last7["para_used"].mean()

    projected_o2 = oxygen_daily_usage * 7
    projected_n95 = n95_daily_usage * 7
    projected_para = para_daily_usage * 7

    latest = df.iloc[-1]
    o2_stock = latest["oxygen_stock"]
    n95_stock = latest["n95_stock"]
    para_stock = latest["para_stock"]

    actions = []

    if o2_stock - projected_o2 < 20:
        actions.append({
            "item": "Oxygen Cylinders",
            "current_stock": int(o2_stock),
            "projected_7day_usage": int(projected_o2),
            "recommendation": "Order 20 oxygen cylinders immediately"
        })

    if n95_stock < 200:
        actions.append({
            "item": "N95 Masks",
            "current_stock": int(n95_stock),
            "projected_7day_usage": int(projected_n95),
            "recommendation": "Order 200 N95 masks"
        })

    if para_stock < 300:
        actions.append({
            "item": "Paracetamol Strips",
            "current_stock": int(para_stock),
            "projected_7day_usage": int(projected_para),
            "recommendation": "Order 300 paracetamol strips"
        })

    if not actions:
        return pd.DataFrame([{
            "item": "All critical items",
            "current_stock": "-",
            "projected_7day_usage": "-",
            "recommendation": "Inventory levels are healthy. No orders needed."
        }])

    return pd.DataFrame(actions)


# ---------------------------------------------------------
# MAIN STREAMLIT APP
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="SwasthyaAI – Hospital Surge Co-pilot", layout="wide")

    st.title("🏥 SwasthyaAI – Hospital Surge & Resource Co-pilot")

    df = load_data()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        horizon_days = st.slider("Forecast horizon (days)", 7, 30, 14)
        show_raw = st.checkbox("Show raw data (last 30 days)", value=False)

        st.markdown("---")
        st.markdown("**Capacity settings**")

        if "cap_mult" not in st.session_state:
            st.session_state.cap_mult = 10

        st.session_state.cap_mult = st.slider(
            "Patients per ER staff per day",
            5, 20, st.session_state.cap_mult
        )

    capacity_multiplier = st.session_state.cap_mult

    # Train ER model
    with st.spinner("Training Prophet model..."):
        model, forecast = train_and_forecast(df, horizon_days=horizon_days)

    # Latest metrics
    latest_er = int(df["er_visits"].iloc[-1])
    next_day_pred = int(forecast.tail(horizon_days).iloc[0]["yhat"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Last observed ER visits", latest_er)
    c2.metric("Predicted ER visits (next day)", next_day_pred)
    c3.metric("Data range", f"{df['date'].min().date()} → {df['date'].max().date()}")

    # -------------------------------
    # ER Forecast Plot
    # -------------------------------
    st.subheader("📈 ER Visit Forecast")

    plot_df = forecast[["ds", "yhat"]]
    plot_df = plot_df.merge(df[["date", "er_visits"]], left_on="ds", right_on="date", how="left")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(plot_df["ds"], plot_df["yhat"])
    ax.scatter(plot_df["ds"], plot_df["er_visits"], s=10, alpha=0.6)
    ax.set_xlabel("Date")
    ax.set_ylabel("ER visits")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ------------------------------------------------------
    # ICU Forecast
    # ------------------------------------------------------
    st.subheader("🛏️ ICU Visit Forecast")

    icu_df = df.rename(columns={"date": "ds", "icu_visits": "y"})
    icu_model = Prophet()
    icu_model.add_regressor("aqi")
    icu_model.add_regressor("temp_c")
    icu_model.add_regressor("festival")
    icu_model.fit(icu_df)

    icu_future = icu_model.make_future_dataframe(periods=horizon_days)
    icu_future["aqi"] = icu_df["aqi"].iloc[-1]
    icu_future["temp_c"] = icu_df["temp_c"].iloc[-1]
    icu_future["festival"] = 0

    icu_forecast = icu_model.predict(icu_future)

    fig2, ax2 = plt.subplots(figsize=(10,4))
    merged = pd.merge(icu_forecast[["ds","yhat"]], df[["date","icu_visits"]],
                      left_on="ds", right_on="date", how="left")
    ax2.plot(merged["ds"], merged["yhat"])
    ax2.scatter(merged["ds"], merged["icu_visits"], s=10)
    st.pyplot(fig2)

    # ------------------------------------------------------
    # OPD Forecast
    # ------------------------------------------------------
    st.subheader("👨‍⚕️ OPD Visit Forecast")

    opd_df = df.rename(columns={"date": "ds", "opd_visits": "y"})
    opd_model = Prophet()
    opd_model.add_regressor("aqi")
    opd_model.add_regressor("temp_c")
    opd_model.add_regressor("festival")
    opd_model.fit(opd_df)

    opd_future = opd_model.make_future_dataframe(periods=horizon_days)
    opd_future["aqi"] = opd_df["aqi"].iloc[-1]
    opd_future["temp_c"] = opd_df["temp_c"].iloc[-1]
    opd_future["festival"] = 0

    opd_forecast = opd_model.predict(opd_future)

    fig3, ax3 = plt.subplots(figsize=(10,4))
    merged2 = pd.merge(opd_forecast[["ds","yhat"]], df[["date","opd_visits"]],
                      left_on="ds", right_on="date", how="left")
    ax3.plot(merged2["ds"], merged2["yhat"])
    ax3.scatter(merged2["ds"], merged2["opd_visits"], s=10)
    st.pyplot(fig3)

    # ------------------------------------------------------
    # AI INSIGHTS (FIRST compute everything)
    # ------------------------------------------------------
    trend_er = forecast["yhat"].tail(7).mean() - forecast["yhat"].tail(14).head(7).mean()
    trend_icu = icu_forecast["yhat"].tail(7).mean() - icu_forecast["yhat"].tail(14).head(7).mean()
    trend_opd = opd_forecast["yhat"].tail(7).mean() - opd_forecast["yhat"].tail(14).head(7).mean()

    # SIMPLE anomaly detection
    last14 = df["er_visits"].tail(14)
    mean14 = last14.mean()
    std14 = last14.std() if last14.std() > 0 else 1
    latest_val = last14.iloc[-1]
    z = (latest_val - mean14) / std14

    # ----------------- INVENTORY (MUST be created BEFORE Executive Summary) ------------------
    inv_df = generate_inventory_recommendations(df)

    # ----------------- STAFFING (also needed for summary) ------------------
    staff_df = generate_staffing_recommendations(forecast, df, capacity_multiplier)

    sev_counts = {
        "Severe Surge (Red)": len(staff_df[staff_df["overload_percent"] > 15]),
        "Moderate (Orange)": len(staff_df[(staff_df["overload_percent"] <= 15) & (staff_df["overload_percent"] > 5)]),
        "Near Capacity (Yellow)": len(staff_df[(staff_df["overload_percent"] <= 5) & (staff_df["overload_percent"] > -5)]),
        "Under Capacity (Green)": len(staff_df[staff_df["overload_percent"] <= -5])
    }

    sev = sev_counts["Severe Surge (Red)"]
    mod = sev_counts["Moderate (Orange)"]
    yel = sev_counts["Near Capacity (Yellow)"]

    # ------------------------------------------------------
    # EXECUTIVE SUMMARY
    # ------------------------------------------------------
    st.subheader("🧠 Executive Summary")

    # inventory summary
    if "Oxygen Cylinders" in inv_df["item"].values:
        inv_msg = "oxygen cylinders are running low"
    elif "N95 Masks" in inv_df["item"].values:
        inv_msg = "N95 mask stock is low"
    elif "Paracetamol Strips" in inv_df["item"].values:
        inv_msg = "paracetamol supplies are below threshold"
    else:
        inv_msg = "All inventory levels are healthy."

    anomaly_msg = f"ER anomaly detected (z={z:.2f})" if abs(z) > 2 else "No ER anomalies detected."

    summary = f"""
    Over the next week:
    - ER demand change: **{trend_er:+.2f}**
    - ICU demand change: **{trend_icu:+.2f}**
    - OPD demand change: **{trend_opd:+.2f}**

    Surge forecast:
    - **{sev} Severe days**, **{mod} Moderate days**, **{yel} Near-capacity days**

    Inventory: {inv_msg}  
    Anomaly Status: {anomaly_msg}
    """

    st.markdown(summary)

    # ------------------------------------------------------
    # SHOW ALL DATAFRAMES
    # ------------------------------------------------------
    st.subheader("🧑‍⚕️ Staffing Recommendations (Next 14 Days)")
    st.dataframe(staff_df, use_container_width=True)

    st.subheader("🔥 Surge Severity Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Severe Surge Days", sev)
    c2.metric("🟠 Moderate Days", mod)
    c3.metric("🟡 Near Capacity", yel)
    c4.metric("🟢 Under Capacity", sev_counts["Under Capacity (Green)"])

    st.subheader("📦 Inventory Recommendations")
    st.dataframe(inv_df, use_container_width=True)

    # ------------------------------------------------------
    # ALERT LOG VIEWER
    # ------------------------------------------------------
    st.subheader("📣 Recent Alerts (from Agent Pipeline)")

    if os.path.exists("alerts_log.json"):
        with open("alerts_log.json", "r", encoding="utf-8") as f:
            alerts = json.load(f)

        if alerts:
            st.dataframe(pd.DataFrame(alerts[-20:]), use_container_width=True)
        else:
            st.info("No alerts yet.")
    else:
        st.info("alerts_log.json not found. Run agent_pipeline.py once.")

    # ------------------------------------------------------
    # RAW DATA
    # ------------------------------------------------------
    if show_raw:
        st.subheader("📄 Raw Data (Last 30 Days)")
        st.dataframe(df.tail(30), use_container_width=True)

    # ------------------------------------------------------
    # DOWNLOAD REPORT
    # ------------------------------------------------------
    st.subheader("📄 Download Daily Report")

    report_html = f"""
    <h1>SwasthyaAI Daily Report</h1>
    <pre>{summary}</pre>
    <h2>Inventory</h2>
    {inv_df.to_html(index=False)}
    <h2>Staffing</h2>
    {staff_df.to_html(index=False)}
    """

    st.download_button("📥 Download Daily Report (HTML)",
                       report_html,
                       "SwasthyaAI_Daily_Report.html",
                       "text/html")


if __name__ == "__main__":
    main()
