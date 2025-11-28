import json
import os
import time
from datetime import datetime

import pandas as pd
from prophet import Prophet

# ---------------- CONFIG ----------------
DATA_PATH = "swasthya_ai_data.csv"
ALERT_LOG_PATH = "alerts_log.json"
FEEDBACK_PATH = "feedback.json"
REPORTS_DIR = "reports"

CAPACITY_MULTIPLIER = 10       # 1 ER staff ≈ 10 patients/day
FORECAST_HORIZON_DAYS = 14
RUN_CONTINUOUSLY = True       # Set True to loop forever
SLEEP_SECONDS = 300            # 5 minutes between runs if looping
# ----------------------------------------


def log(message, agent="SYSTEM"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] [{agent}] {message}"
    print(line)


def ensure_files():
    """Create folders/files if missing."""
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR, exist_ok=True)

    if not os.path.exists(ALERT_LOG_PATH):
        with open(ALERT_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)

    if not os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)


def load_data(path=DATA_PATH):
    log("Loading hospital data...", agent="DataMonitorAgent")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    log(f"Data loaded. Date range: {df['date'].min().date()} → {df['date'].max().date()}",
        agent="DataMonitorAgent")
    return df


def run_forecast_for_column(df, column_name, horizon_days=FORECAST_HORIZON_DAYS):
    """Train Prophet model on a specified column (er_visits, icu_visits, opd_visits)."""
    log(f"Training Prophet for {column_name}...", agent="ForecastAgent")

    df_prophet = df.rename(columns={"date": "ds", column_name: "y"})
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
    upcoming = forecast.tail(horizon_days)

    min_pred = int(upcoming["yhat"].min())
    max_pred = int(upcoming["yhat"].max())
    log(f"{column_name}: forecast {horizon_days} days range = {min_pred}–{max_pred}",
        agent="ForecastAgent")

    return forecast, upcoming


def detect_anomaly(df, column_name="er_visits", window=14, threshold_std=2.0):
    """Simple anomaly: last value deviates > threshold_std * std from rolling mean."""
    # compute model residuals (if Prophet available)
    try:
        from prophet import Prophet
        df_prop = df.rename(columns={'date':'ds', column_name:'y'})
        m = Prophet()
        m.add_regressor('aqi'); m.add_regressor('temp_c'); m.add_regressor('festival')
        m.fit(df_prop)
        pred = m.predict(df_prop[['ds']])
        resid = df[column_name] - pred['yhat']
        last_val = resid.iloc[-1]
        prev = resid.iloc[-(window+1):-1]
        mean = prev.mean()
        std = prev.std() if prev.std() > 0 else 1e-6
        z_score = (last_val - mean) / std
    except Exception:
    # fallback to simple approach
        series = df[column_name].tail(window + 1)
        last_val = series.iloc[-1]
        prev = series.iloc[:-1]
        mean = prev.mean(); std = prev.std() if prev.std()>0 else 1e-6
        z_score = (last_val - mean) / std


    if abs(z_score) >= threshold_std:
        level = "high" if z_score > 0 else "low"
        log(f"Anomaly detected in {column_name}: last={last_val}, mean={mean:.1f}, z={z_score:.2f}",
            agent="ForecastAgent")
        return {
            "metric": column_name,
            "last_value": int(last_val),
            "mean": round(mean, 1),
            "z_score": round(z_score, 2),
            "level": level
        }
    return None


def classify_surge(predicted, capacity):
    """Return color + label based on overload percent."""
    overload = (predicted - capacity) / capacity * 100
    if overload < -5:
        return "green", overload, "Normal / Under capacity"
    elif overload < 5:
        return "yellow", overload, "Close to capacity"
    elif overload < 15:
        return "orange", overload, "Moderate surge"
    else:
        return "red", overload, "Severe surge"


def generate_staffing_recommendations(er_forecast_upcoming, df, capacity_multiplier=CAPACITY_MULTIPLIER):
    log("Computing staffing recommendations...", agent="OpsPlannerAgent")

    latest = df.iloc[-1]
    er_staff = latest["er_staff_capacity"]
    er_capacity = er_staff * capacity_multiplier

    recs = []
    for _, row in er_forecast_upcoming.iterrows():
        predicted = row["yhat"]
        date = row["ds"].date()

        color, overload, label = classify_surge(predicted, er_capacity)

        if color == "green":
            action = "No action needed"
        elif color == "yellow":
            action = "Monitor staffing and be prepared to add 1 nurse"
        elif color == "orange":
            action = "Add 1 extra nurse to ER shift"
        else:  # red
            action = "Add 1 nurse + 1 medical officer, activate surge protocol"

        recs.append({
            "type": "staffing",
            "date": str(date),
            "predicted_er_visits": int(round(predicted)),
            "er_capacity_patients_per_day": int(er_capacity),
            "overload_percent": round(overload, 1),
            "severity_color": color,
            "severity_label": label,
            "recommended_action": action,
        })

    reds = [r for r in recs if r["severity_color"] == "red"]
    oranges = [r for r in recs if r["severity_color"] == "orange"]
    log(f"Staffing plan: severe={len(reds)}, moderate={len(oranges)} surge days.",
        agent="OpsPlannerAgent")

    return recs


def generate_inventory_recommendations(df):
    log("Evaluating inventory for next 7 days...", agent="OpsPlannerAgent")

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
            "type": "inventory",
            "item": "Oxygen Cylinders",
            "current_stock": int(o2_stock),
            "projected_7day_usage": int(projected_o2),
            "recommendation": "Order 20 oxygen cylinders immediately"
        })

    if n95_stock < 200:
        actions.append({
            "type": "inventory",
            "item": "N95 Masks",
            "current_stock": int(n95_stock),
            "projected_7day_usage": int(projected_n95),
            "recommendation": "Order 200 N95 masks"
        })

    if para_stock < 300:
        actions.append({
            "type": "inventory",
            "item": "Paracetamol Strips",
            "current_stock": int(para_stock),
            "projected_7day_usage": int(projected_para),
            "recommendation": "Order 300 paracetamol strips"
        })

    log(f"Inventory actions needed: {len(actions)}", agent="OpsPlannerAgent")
    return actions


# ----------- ALERT SENDING (STUBS + REAL-HOOK READY) -----------

def send_whatsapp_alert(to_number: str, message: str):
    """
    Stub for WhatsApp alert.
    To enable real sending:
      - Create a WhatsApp Cloud API app on Meta
      - Get access token, phone_number_id
      - Use 'requests' to call the Graph API endpoint
    For now we just log the message.
    """
    log(f"[WhatsApp → {to_number}] {message}", agent="ExecutionAgent")


def send_sms_alert(to_number: str, message: str):
    """
    Stub for SMS alert.
    For real sending:
      - pip install twilio
      - from twilio.rest import Client
      - client = Client(ACCOUNT_SID, AUTH_TOKEN)
      - client.messages.create(to=to_number, from_='YOUR_TWILIO_NUMBER', body=message)
    For demo we only log.
    """
    log(f"[SMS → {to_number}] {message}", agent="ExecutionAgent")


def append_alert_to_log(alert_obj):
    """Save alerts to a JSON file so Streamlit can read them."""
    try:
        with open(ALERT_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = []

    data.append(alert_obj)

    with open(ALERT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_daily_report(staff_recs, inventory_recs, anomaly_info):
    """Write a simple HTML report file. Can be opened & 'Print to PDF'."""
    today_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(REPORTS_DIR, f"SwasthyaAI_Report_{today_str}.html")

    html = ["<html><body>"]
    html.append(f"<h1>SwasthyaAI Daily Ops Report – {today_str}</h1>")

    # Anomaly
    html.append("<h2>Anomaly Detection</h2>")
    if anomaly_info:
        html.append(
            f"<p>Anomaly in {anomaly_info['metric']}: "
            f"last={anomaly_info['last_value']}, "
            f"mean={anomaly_info['mean']}, "
            f"z={anomaly_info['z_score']}, "
            f"level={anomaly_info['level']}</p>"
        )
    else:
        html.append("<p>No major anomalies detected.</p>")

    # Staffing
    html.append("<h2>Staffing Recommendations</h2><ul>")
    for r in staff_recs:
        html.append(
            f"<li>{r['date']}: predicted {r['predicted_er_visits']} ER visits "
            f"(capacity {r['er_capacity_patients_per_day']}, "
            f"overload {r['overload_percent']}%, "
            f"severity {r['severity_label']}) – "
            f"<b>{r['recommended_action']}</b></li>"
        )
    html.append("</ul>")

    # Inventory
    html.append("<h2>Inventory Recommendations</h2><ul>")
    if inventory_recs:
        for i in inventory_recs:
            html.append(
                f"<li>{i['item']}: stock {i['current_stock']}, "
                f"projected 7-day usage {i['projected_7day_usage']} – "
                f"<b>{i['recommendation']}</b></li>"
            )
    else:
        html.append("<li>No inventory actions needed.</li>")
    html.append("</ul></body></html>")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    log(f"Daily HTML report generated at {report_path}", agent="ExecutionAgent")


def execute_alerts(staff_recs, inventory_recs, anomaly_info):
    log("Translating recommendations into alerts...", agent="ExecutionAgent")

    # You can configure target numbers / WhatsApp recipients here:
    er_head = "+911234567890"
    nursing_supervisor = "+919876543210"
    store_manager = "+911112223334"

    # STAFF ALERTS
    for r in staff_recs:
        action = r["recommended_action"]
        if action.startswith("No action needed"):
            continue

        msg = (
            f"[STAFFING ALERT] {r['date']}: "
            f"Predicted {r['predicted_er_visits']} ER visits "
            f"(capacity {r['er_capacity_patients_per_day']}, "
            f"overload {r['overload_percent']}%, "
            f"severity={r['severity_label']}). "
            f"Action: {action}"
        )

        alert_obj = {
            "timestamp": datetime.now().isoformat(),
            "channel": "whatsapp+sms",
            "category": "staffing",
            "date": r["date"],
            "message": msg,
        }
        append_alert_to_log(alert_obj)

        send_whatsapp_alert(er_head, msg)
        send_whatsapp_alert(nursing_supervisor, msg)
        send_sms_alert(er_head, msg)

    # INVENTORY ALERTS
    for i in inventory_recs:
        msg = (
            f"[INVENTORY ALERT] {i['item']}: stock {i['current_stock']}, "
            f"projected 7-day usage {i['projected_7day_usage']}. "
            f"Action: {i['recommendation']}"
        )

        alert_obj = {
            "timestamp": datetime.now().isoformat(),
            "channel": "whatsapp+sms",
            "category": "inventory",
            "item": i["item"],
            "message": msg,
        }
        append_alert_to_log(alert_obj)

        send_whatsapp_alert(store_manager, msg)
        send_sms_alert(store_manager, msg)

    # Anomaly alert (info-level)
    if anomaly_info:
        msg = (
            f"[ANOMALY] {anomaly_info['metric']}: last="
            f"{anomaly_info['last_value']}, mean={anomaly_info['mean']}, "
            f"z={anomaly_info['z_score']}, level={anomaly_info['level']}"
        )
        alert_obj = {
            "timestamp": datetime.now().isoformat(),
            "channel": "log-only",
            "category": "anomaly",
            "metric": anomaly_info["metric"],
            "message": msg,
        }
        append_alert_to_log(alert_obj)
        log(msg, agent="ExecutionAgent")

    if not staff_recs and not inventory_recs and not anomaly_info:
        log("No alerts to send. System in green state.", agent="ExecutionAgent")


def run_once():
    log("Starting SwasthyaAI autonomous ops run...", agent="SYSTEM")
    ensure_files()

    # 1) Load data
    df = load_data()

    # 2) Forecasts
    er_forecast, er_upcoming = run_forecast_for_column(df, "er_visits")
    icu_forecast, icu_upcoming = run_forecast_for_column(df, "icu_visits")
    opd_forecast, opd_upcoming = run_forecast_for_column(df, "opd_visits")

    # 3) Anomaly detection
    anomaly_info = detect_anomaly(df, "er_visits")

    # 4) Ops planning
    staff_recs = generate_staffing_recommendations(er_upcoming, df)
    inventory_recs = generate_inventory_recommendations(df)

    # 5) Report + Alerts
    generate_daily_report(staff_recs, inventory_recs, anomaly_info)
    execute_alerts(staff_recs, inventory_recs, anomaly_info)

    log("Run complete. SwasthyaAI is ready for next cycle.", agent="SYSTEM")


def main():
    if RUN_CONTINUOUSLY:
        while True:
            run_once()
            log(f"Sleeping for {SLEEP_SECONDS} seconds before next run...", agent="SYSTEM")
            time.sleep(SLEEP_SECONDS)
    else:
        run_once()


if __name__ == "__main__":
    main()
