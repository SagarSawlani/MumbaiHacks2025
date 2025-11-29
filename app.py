# app.py — Final cleaned & fixed version (with "Run pipeline now" feature)
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import os, json, textwrap
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ----------------- RUN PIPELINE IMPORTS (safe) -----------------
import threading
import time as _time
import glob
import html as _html

_pipeline_import_error = None
pipeline_run_once = None
try:
    from agent_pipeline import run_once as pipeline_run_once
except Exception as e:
    _pipeline_import_error = str(e)
    pipeline_run_once = None

# ----------------- HELPERS -----------------
def build_calendar_heatmap_from_forecast(forecast, days=30, value_col='yhat'):
    """
    Build a weeks x weekdays matrix and a Plotly heatmap figure from Prophet forecast.
    - forecast: DataFrame with columns 'ds' (datetime) and value_col (yhat).
    - days: how many upcoming days to show (int).
    - returns: plotly.graph_objects.Figure
    """
    import pandas as pd
    import numpy as np

    # Prepare df with only needed columns and next `days`
    df = forecast[['ds', value_col]].copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    df = df.tail(days).reset_index(drop=True)  # last N days (should be future portion)

    if df.empty:
        raise ValueError("Forecast dataframe is empty for heatmap")

    # Week index starting from 0 for the first date in this slice
    first_date = df['ds'].dt.normalize().min()
    df['days_from_start'] = (df['ds'].dt.normalize() - first_date).dt.days
    df['week'] = (df['days_from_start'] // 7).astype(int)
    df['weekday'] = df['ds'].dt.weekday  # 0=Mon .. 6=Sun

    # Pivot to create weeks x weekdays matrix
    pivot_values = df.pivot(index='week', columns='weekday', values=value_col)
    # Reindex columns to ensure all weekdays 0..6 exist
    pivot_values = pivot_values.reindex(columns=range(7))

    # Construct matrix of date labels for hover/customdata
    pivot_dates = df.pivot(index='week', columns='weekday', values='ds')
    pivot_dates = pivot_dates.reindex(columns=range(7))

    # Convert pivot to numpy arrays for plotting (replace missing with np.nan)
    z = pivot_values.values.astype(float)
    # Create string matrix for hover (dates), keep empty string for NaN cells
    customdata = np.full(z.shape, "", dtype=object)
    for i, wk in enumerate(pivot_values.index):
        for j in range(7):
            val = pivot_dates.loc[wk, j] if (wk in pivot_dates.index and j in pivot_dates.columns) else None
            if pd.notna(val):
                customdata[i, j] = pd.to_datetime(val).date().isoformat()
            else:
                customdata[i, j] = ""

    # Labels for axes
    weekday_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    week_labels = [f"Week {int(wk)+1}" for wk in pivot_values.index]

    # Build Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=weekday_labels,
        y=week_labels,
        text=np.round(z, 0),
        hoverinfo='text',
        customdata=customdata,
        hovertemplate=
            "<b>%{customdata}</b><br>" +
            "Week: %{y}<br>" +
            "Weekday: %{x}<br>" +
            "Predicted visits: %{z:.0f}<extra></extra>",
        colorscale='YlOrRd',
        colorbar=dict(title='Predicted visits'),
        zmin=np.nanmin(z) if not np.all(np.isnan(z)) else 0,
        zmax=np.nanmax(z) if not np.all(np.isnan(z)) else 1,
        showscale=True
    ))

    fig.update_layout(
        title=f"Patient load heatmap — next {len(df)} days",
        xaxis_title="Weekday",
        yaxis_title="Week (rolling window)",
        yaxis_autorange='reversed',  # have earliest week at top
        margin=dict(t=50, l=120, r=40, b=40),
        template='plotly_dark'
    )

    # If lots of weeks, increase height
    fig.update_layout(height=200 + 60 * len(week_labels))

    # Annotate cells with numbers (optional) — handled by text arg above
    return fig

def future_regressor_by_dayofyear(df, col, future_df):
    tmp = df.copy()
    tmp['doy'] = tmp['date'].dt.dayofyear
    doy_avg = tmp.groupby('doy')[col].mean()
    future = future_df.copy()
    future['doy'] = future['ds'].dt.dayofyear
    future[col] = future['doy'].map(doy_avg).fillna(df[col].iloc[-1])
    return future[col].values

@st.cache_data
def train_and_forecast_with_regressors(df, target_col='er_visits', horizon_days: int = 14):
    df_prophet = df.rename(columns={"date": "ds", target_col: "y"})
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    festivals = df[df.get('festival', 0) == 1][['date']].rename(columns={'date': 'ds'})
    holidays = None
    if not festivals.empty:
        holidays = festivals.assign(holiday='local_fest')

    m = Prophet(holidays=holidays, yearly_seasonality=True, weekly_seasonality=True,
                changepoint_prior_scale=0.05)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    for r in ['aqi', 'temp_c', 'festival']:
        if r in df_prophet.columns:
            m.add_regressor(r)

    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=horizon_days)
    if 'aqi' in df.columns:
        future['aqi'] = future_regressor_by_dayofyear(df, 'aqi', future)
    if 'temp_c' in df.columns:
        future['temp_c'] = future_regressor_by_dayofyear(df, 'temp_c', future)
    future['festival'] = 0

    forecast = m.predict(future)

    meta = {
        'trained_on_rows': len(df),
        'history_start': df['date'].min().date(),
        'history_end': df['date'].max().date()
    }
    return m, forecast, meta

def safe_get_uncertainty(forecast_df, model, hist_df, target_col):
    if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
        return forecast_df["yhat_lower"].copy(), forecast_df["yhat_upper"].copy()

    try:
        hist_pred = model.predict(hist_df.rename(columns={'date':'ds'}))
        if 'yhat' in hist_pred.columns and target_col in hist_df.columns:
            resid = hist_df[target_col].values - hist_pred['yhat'].values
            sigma = float(np.nanstd(resid)) if np.nanstd(resid) > 0 else 1.0
        else:
            sigma = 1.0
    except Exception:
        sigma = 1.0

    lower = forecast_df["yhat"] - 1.96 * sigma
    upper = forecast_df["yhat"] + 1.96 * sigma
    return lower, upper

# ----------------- DATA -----------------
@st.cache_data
def load_data(path="swasthya_ai_data.csv"):
    df = pd.read_csv(path)
    if 'date' not in df.columns:
        raise ValueError("CSV must have a 'date' column")
    df["date"] = pd.to_datetime(df["date"])
    return df

# ----------------- ENGINES -----------------
def generate_staffing_recommendations(forecast, df, capacity_multiplier):
    forecast_14 = forecast[["ds", "yhat"]].tail(14).copy()
    forecast_14["ds"] = pd.to_datetime(forecast_14["ds"])

    latest = df.iloc[-1]
    er_staff = int(latest.get("er_staff_capacity", 0))
    er_capacity = er_staff * capacity_multiplier

    recs = []
    for _, row in forecast_14.iterrows():
        predicted = float(row["yhat"])
        date = row["ds"].date()
        overload = (predicted - er_capacity) / er_capacity * 100 if er_capacity > 0 else 0.0

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
            "date": str(date),
            "predicted_er_visits": int(round(predicted)),
            "er_capacity_patients_per_day": int(er_capacity),
            "overload_percent": round(overload, 1),
            "recommended_action": action
        })

    return pd.DataFrame(recs)

def generate_inventory_recommendations(df):
    last7 = df.tail(7)

    oxygen_daily_usage = float(last7["oxygen_used"].mean()) if "oxygen_used" in last7.columns else 0.0
    n95_daily_usage = float(last7["n95_used"].mean()) if "n95_used" in last7.columns else 0.0
    para_daily_usage = float(last7["para_used"].mean()) if "para_used" in last7.columns else 0.0

    projected_o2 = oxygen_daily_usage * 7
    projected_n95 = n95_daily_usage * 7
    projected_para = para_daily_usage * 7

    latest = df.iloc[-1]
    o2_stock = int(latest.get("oxygen_stock", 0))
    n95_stock = int(latest.get("n95_stock", 0))
    para_stock = int(latest.get("para_stock", 0))

    actions = []
    if o2_stock - projected_o2 < 20:
        actions.append({
            "item": "Oxygen Cylinders",
            "current_stock": int(o2_stock),
            "projected_7day_usage": int(round(projected_o2)),
            "recommendation": "Order 20 oxygen cylinders immediately"
        })
    if n95_stock < 200:
        actions.append({
            "item": "N95 Masks",
            "current_stock": int(n95_stock),
            "projected_7day_usage": int(round(projected_n95)),
            "recommendation": "Order 200 N95 masks"
        })
    if para_stock < 300:
        actions.append({
            "item": "Paracetamol Strips",
            "current_stock": int(para_stock),
            "projected_7day_usage": int(round(projected_para)),
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

# ---------------- WHAT-IF SIMULATOR HELPERS -----------------
def build_simulated_future_from_model(model, df, horizon_days, sim_aqi, sim_temp, sim_festival_days=None):
    future = model.make_future_dataframe(periods=horizon_days)
    future = future.sort_values("ds").reset_index(drop=True)

    last_aqi = df["aqi"].iloc[-1] if "aqi" in df.columns else sim_aqi
    last_temp = df["temp_c"].iloc[-1] if "temp_c" in df.columns else sim_temp

    future["aqi"] = sim_aqi if sim_aqi is not None else last_aqi
    future["temp_c"] = sim_temp if sim_temp is not None else last_temp
    future["festival"] = 0

    if sim_festival_days:
        sim_fest_set = set(pd.to_datetime(sim_festival_days).normalize())
        future["festival"] = future["ds"].dt.normalize().isin(sim_fest_set).astype(int)

    future["ds"] = pd.to_datetime(future["ds"])
    return model.predict(future), future

def compare_forecasts_and_staffing(base_forecast, sim_forecast, df, capacity_multiplier):
    base_14 = base_forecast[["ds","yhat"]].tail(14).reset_index(drop=True)
    sim_14 = sim_forecast[["ds","yhat"]].tail(14).reset_index(drop=True)

    metrics = {
        "base_mean": float(base_14["yhat"].mean()),
        "sim_mean": float(sim_14["yhat"].mean()),
        "base_min": int(base_14["yhat"].min()),
        "sim_min": int(sim_14["yhat"].min()),
        "base_max": int(base_14["yhat"].max()),
        "sim_max": int(sim_14["yhat"].max()),
        "delta_mean": float(sim_14["yhat"].mean() - base_14["yhat"].mean())
    }

    base_staff = generate_staffing_recommendations(base_forecast, df, capacity_multiplier)
    sim_staff = generate_staffing_recommendations(sim_forecast, df, capacity_multiplier)
    return metrics, base_staff, sim_staff
# ---------------- END WHAT-IF HELPERS -----------------

def plot_summary_card_html(title, last_observed, next_pred, upcoming, extra_text=None, actions=None, font_px=20):
    upcoming = upcoming.copy()
    min_pred = int(upcoming["yhat"].min())
    max_pred = int(upcoming["yhat"].max())
    mean_pred = float(upcoming["yhat"].mean())

    actions_html = ""
    if actions:
        for a in actions:
            # keep it plain text inside list item
            actions_html += f"<li>{str(a)}</li>"

    extra = (extra_text or "").replace("<", "&lt;").replace(">", "&gt;")

    html = f"""
    <div style="text-align:left; font-size:{font_px}px; line-height:1.35; margin-bottom:10px;">
      <strong style="font-size:{int(font_px*1.05)}px;">{title} — Quick snapshot (next {len(upcoming)} days)</strong>
      <div style="margin-top:8px;">
        <div><strong>Pred range:</strong> {min_pred} → {max_pred} (mean {mean_pred:.1f})</div>
        <div style="margin-top:6px;">{extra}</div>
        <div style="margin-top:8px;"><strong>Suggested quick actions:</strong></div>
        <ul style="margin-top:6px; margin-left:20px;">
          {actions_html}
        </ul>
      </div>
      <div style="margin-top:8px;"><small>Last observed: <strong>{int(last_observed)}</strong> · Next day pred: <strong>{int(next_pred)}</strong></small></div>
    </div>
    """
    clean_html = html.replace("</div></div>", "</div>").replace("</div> </div>", "</div>")
    st.markdown(clean_html, unsafe_allow_html=True)

# ----------------- MAIN -----------------
def main():
    st.set_page_config(page_title="SwasthyaAI – Hospital Surge Co-pilot", layout="wide")
    st.title("🏥 SwasthyaAI – Hospital Surge & Resource Co-pilot")

    df = load_data()

    with st.sidebar:
        st.header("⚙️ Controls")
        horizon_days = st.slider("Forecast horizon (days)", 7, 30, 14)
        show_raw = st.checkbox("Show raw data (last 30 days)", value=False)
        st.markdown("---")
        st.markdown("**Capacity settings**")
        if "cap_mult" not in st.session_state:
            st.session_state.cap_mult = 10
        st.session_state.cap_mult = st.slider("Patients per ER staff per day", 5, 20, st.session_state.cap_mult)

        # NEW: calendar heatmap days selector
        st.markdown("---")
        st.markdown("📅 Calendar heatmap")
        if "heatmap_days" not in st.session_state:
            st.session_state.heatmap_days = 30
        st.session_state.heatmap_days = st.slider("Heatmap horizon (days)", 7, 60, st.session_state.heatmap_days)

        # ----------------- AGENT CONTROLS (Manual run) -----------------
        st.markdown("---")
        st.subheader("⚙️ Agent Controls")

        if pipeline_run_once is None:
            st.info("agent_pipeline.run_once() not available in this environment.")
            if _pipeline_import_error:
                st.caption(_html.escape(_pipeline_import_error))
        else:
            if st.button("▶️ Run pipeline now"):
                # set a simple session flag (optional)
                st.session_state["_pipeline_running"] = True
                try:
                    with st.spinner("Running agent pipeline (this may take a minute)..."):
                        pipeline_run_once()
                    st.success("Pipeline run finished. Check Recent Alerts and reports/ directory.")
                except Exception as err:
                    st.error(f"Pipeline run failed: {err}")
                finally:
                    st.session_state["_pipeline_running"] = False

            # show latest report if available
            latest_reports = sorted(glob.glob("reports/*.html"), key=os.path.getmtime) if os.path.exists("reports") else []
            if latest_reports:
                latest = latest_reports[-1]
                mtime = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(os.path.getmtime(latest)))
                st.markdown(f"**Latest report:** { os.path.basename(latest) }  \n_{mtime}_")
                try:
                    with open(latest, "r", encoding="utf-8") as f:
                        html_bytes = f.read()
                    st.download_button("📥 Download latest report", html_bytes, file_name=os.path.basename(latest), mime="text/html")
                except Exception:
                    st.caption("Could not prepare report download.")
            else:
                st.info("No HTML reports found. Run the pipeline to generate one.")
        # ----------------- END AGENT CONTROLS -----------------

    capacity_multiplier = st.session_state.cap_mult

    # ---------------- ER MODEL & 2x3 SUBPLOT LAYOUT ----------------
    with st.spinner("Training Prophet model for ER..."):
        model, forecast, meta = train_and_forecast_with_regressors(df, 'er_visits', horizon_days=horizon_days)
    st.info(f"ER model trained on {meta['trained_on_rows']} days: {meta['history_start']} → {meta['history_end']}")
    st.subheader("📊 ER")

    full_forecast = forecast.copy()
    plot_df = full_forecast[["ds", "yhat"]].merge(df[["date", "er_visits"]], left_on="ds", right_on="date", how="left")
    plot_df["ds"] = pd.to_datetime(plot_df["ds"])

    yhat_lower, yhat_upper = safe_get_uncertainty(full_forecast, model, df, 'er_visits')

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14,12))
    axes = axes.flatten()

    # Panel 1
    ax = axes[0]
    ax.plot(plot_df["ds"], plot_df["yhat"], label="Predicted (yhat)")
    try:
        lower_al = pd.Series(yhat_lower.values, index=full_forecast['ds']).reindex(plot_df['ds']).values
        upper_al = pd.Series(yhat_upper.values, index=full_forecast['ds']).reindex(plot_df['ds']).values
        ax.fill_between(plot_df["ds"], lower_al, upper_al, alpha=0.12)
    except Exception:
        ax.fill_between(plot_df["ds"], plot_df["yhat"]*0.9, plot_df["yhat"]*1.1, alpha=0.08)
    ax.scatter(plot_df["ds"], plot_df["er_visits"], s=8, alpha=0.6, label="Actual")
    ax.set_title("Full forecast vs actual")
    ax.set_xlabel("")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2
    ax = axes[1]
    last_mask = plot_df["ds"] >= (plot_df["ds"].max() - pd.Timedelta(days=30))
    ax.plot(plot_df.loc[last_mask,"ds"], plot_df.loc[last_mask,"yhat"], label="Predicted (last 30d)")
    ax.scatter(plot_df.loc[last_mask,"ds"], plot_df.loc[last_mask,"er_visits"], s=10, alpha=0.7)
    ax.set_title("Zoom: last 30 days")
    ax.set_xlabel("")
    ax.grid(alpha=0.3)

    # Panel 3
    ax = axes[2]
    up = full_forecast[["ds","yhat"]].tail(horizon_days)
    try:
        lower_up = yhat_lower.tail(horizon_days).values if hasattr(yhat_lower, "values") else np.array(yhat_lower[-horizon_days:])
        upper_up = yhat_upper.tail(horizon_days).values if hasattr(yhat_upper, "values") else np.array(yhat_upper[-horizon_days:])
        ax.plot(up["ds"], up["yhat"], label="Upcoming preds")
        ax.fill_between(up["ds"], lower_up, upper_up, alpha=0.12)
    except Exception:
        ax.plot(up["ds"], up["yhat"])
        ax.fill_between(up["ds"], up["yhat"]*0.9, up["yhat"]*1.1, alpha=0.06)
    ax.set_title(f"Upcoming {horizon_days} days (uncertainty)")
    ax.set_xlabel("")
    ax.grid(alpha=0.3)

    # Panel 4 (trend)
    ax = axes[3]
    if 'trend' in full_forecast.columns:
        ax.plot(full_forecast["ds"], full_forecast["trend"], label="Trend")
        ax.set_title("Estimated trend")
    else:
        ax.text(0.5,0.5,"Trend not available", ha='center', va='center')
    ax.grid(alpha=0.3)

    # Panel 5 (residuals)
    ax = axes[4]
    try:
        hist_pred = model.predict(df.rename(columns={'date':'ds'}))
        df['yhat_hist'] = hist_pred['yhat'].values
        df['resid'] = df['er_visits'] - df['yhat_hist']
        ax.plot(df['date'], df['resid'])
        ax.axhline(0, color='k', linestyle='--', alpha=0.6)
        ax.set_title("Residuals (history)")
    except Exception:
        ax.text(0.5,0.5,"Residuals not computed", ha='center', va='center')
    ax.grid(alpha=0.3)

    # Panel 6 (resid dist)
    ax = axes[5]
    try:
        ax.hist(df['resid'].dropna(), bins=20)
        ax.set_title("Residuals distribution")
    except Exception:
        ax.text(0.5,0.5,"No residuals to show", ha='center', va='center')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # ----------------- ER calendar heatmap (NEW) -----------------
    try:
        # get slider value from session state (falls back to 30)
        heatmap_days = int(st.session_state.get("heatmap_days", 30))
        # build heatmap from the full forecast (uses 'yhat' by default)
        heatmap_fig = build_calendar_heatmap_from_forecast(full_forecast, days=heatmap_days, value_col='yhat')
        st.subheader(f"📅 ER load calendar (next {heatmap_days} days)")
        st.write("Color intensity shows predicted ER visits. Hover a cell for date + predicted value.")
        st.plotly_chart(heatmap_fig, use_container_width=True)
    except Exception as e:
        # don't break the app if heatmap fails; show a helpful warning
        st.warning(f"Could not render calendar heatmap: {e}")
    # ----------------- end heatmap -----------------

    # ER summary
    er_upcoming = full_forecast[["ds","yhat"]].tail(horizon_days)
    latest_er = int(df["er_visits"].iloc[-1]) if "er_visits" in df.columns else 0
    future_slice = full_forecast[full_forecast['ds'] > df['date'].max()]
    if not future_slice.empty:
        next_day_er = int(round(float(future_slice.iloc[0]["yhat"])))
    else:
        next_day_er = int(round(float(er_upcoming.iloc[0]["yhat"])))
    er_extra = "ER model: weekly/yearly seasonality + festival flags"
    er_actions = [
        "If predicted > capacity: Be ready to Add staff.",
        "Keep a check on Oxygen/N95 stock if ICU load increases."
    ]
    plot_summary_card_html("ER", latest_er, next_day_er, er_upcoming, extra_text=er_extra, actions=er_actions, font_px=22)

    # ----------------- WHAT-IF SIMULATOR UI (in main, after model exists) -----------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔮 What-If Simulator (ER)")

    default_aqi = int(df["aqi"].mean()) if "aqi" in df.columns else 100
    default_temp = int(df["temp_c"].mean()) if "temp_c" in df.columns else 25

    sim_aqi = st.sidebar.slider("Simulated future AQI", 10, 500, default_aqi)
    sim_temp = st.sidebar.slider("Simulated future Temp (°C)", 5, 45, default_temp)
    sim_festival_toggle = st.sidebar.checkbox("Add a festival day in horizon?", value=False)

    sim_festival_dates = None
    if sim_festival_toggle:
        sim_date = st.sidebar.date_input("Festival date (pick 1 within horizon)", value=df["date"].max().date())
        sim_festival_dates = [sim_date]

    if st.sidebar.button("🔁 Simulate ER (Apply)"):
        with st.spinner("Running what-if simulation..."):
            try:
                sim_forecast, sim_future = build_simulated_future_from_model(model, df, horizon_days, sim_aqi, sim_temp, sim_festival_dates)
                metrics, base_staff, sim_staff = compare_forecasts_and_staffing(forecast, sim_forecast, df, capacity_multiplier)

                st.info(f"Simulation applied — Δmean = {metrics['delta_mean']:+.2f} (sim_mean {metrics['sim_mean']:.1f} vs base_mean {metrics['base_mean']:.1f})")
                st.markdown(f"**Range (base):** {metrics['base_min']} → {metrics['base_max']}  \n**Range (sim):** {metrics['sim_min']} → {metrics['sim_max']}")

                sim_plot_df = pd.DataFrame({
                    "ds": sim_forecast["ds"].tail(horizon_days).values,
                    "base_yhat": forecast["yhat"].tail(horizon_days).values,
                    "sim_yhat": sim_forecast["yhat"].tail(horizon_days).values
                })
                fig_sim, ax_sim = plt.subplots(figsize=(10,3))
                ax_sim.plot(sim_plot_df["ds"], sim_plot_df["base_yhat"], label="Base forecast")
                ax_sim.plot(sim_plot_df["ds"], sim_plot_df["sim_yhat"], label="Simulated forecast", linestyle="--")
                ax_sim.set_title("Base vs Simulated ER forecast (next days)")
                ax_sim.set_xlabel("Date")
                ax_sim.set_ylabel("ER visits")
                ax_sim.legend()
                ax_sim.grid(alpha=0.3)
                st.pyplot(fig_sim)

                st.subheader("Staffing: Base vs Simulated (next 14 days)")
                col_b, col_s = st.columns(2)
                with col_b:
                    st.markdown("**Base staffing recommendations**")
                    st.dataframe(base_staff, use_container_width=True, height=320)
                with col_s:
                    st.markdown("**Simulated staffing recommendations**")
                    st.dataframe(sim_staff, use_container_width=True, height=320)

            except Exception as e:
                st.error(f"Simulation failed: {e}")
    # ----------------- END WHAT-IF SIMULATOR -----------------

    # ---------------- ICU ----------------
    st.subheader("🛏️ ICU — compact view")
    with st.spinner("Training Prophet model for ICU..."):
        icu_model, icu_forecast, icu_meta = train_and_forecast_with_regressors(df, 'icu_visits', horizon_days=horizon_days)
    st.info(f"ICU model trained on {icu_meta['trained_on_rows']} days")

    fig2, ax2 = plt.subplots(figsize=(8,3))
    merged = pd.merge(icu_forecast[["ds","yhat"]], df[["date","icu_visits"]], left_on="ds", right_on="date", how="left")
    ax2.plot(merged["ds"], merged["yhat"], label="Predicted")
    icu_lower, icu_upper = safe_get_uncertainty(icu_forecast, icu_model, df, 'icu_visits')
    try:
        lower_al = pd.Series(icu_lower.values, index=icu_forecast['ds']).reindex(merged['ds']).values
        upper_al = pd.Series(icu_upper.values, index=icu_forecast['ds']).reindex(merged['ds']).values
        ax2.fill_between(merged["ds"], lower_al, upper_al, alpha=0.08)
    except Exception:
        ax2.fill_between(merged["ds"], merged["yhat"]*0.9, merged["yhat"]*1.1, alpha=0.06)
    ax2.scatter(merged["ds"], merged["icu_visits"], s=8, label="Actual")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("ICU visits")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    icu_upcoming = icu_forecast[["ds","yhat"]].tail(horizon_days)
    latest_icu = int(df["icu_visits"].iloc[-1]) if "icu_visits" in df.columns else 0
    future_icu = icu_forecast[icu_forecast['ds'] > df['date'].max()]
    next_day_icu = int(round(float(future_icu.iloc[0]["yhat"]))) if not future_icu.empty else int(round(float(icu_upcoming.iloc[0]["yhat"])))
    icu_extra = "ICU may have uncertainity, observe the upper band properly."
    icu_actions = [
        "Check Ventilator / ICU bed availability.",
        "Critical-care staff roster review."
    ]
    plot_summary_card_html("ICU", latest_icu, next_day_icu, icu_upcoming, extra_text=icu_extra, actions=icu_actions, font_px=18)

    # ---------------- OPD ----------------
    st.subheader("👨‍⚕️ OPD — compact view")
    with st.spinner("Training Prophet model for OPD..."):
        opd_model, opd_forecast, opd_meta = train_and_forecast_with_regressors(df, 'opd_visits', horizon_days=horizon_days)
    st.info(f"OPD model trained on {opd_meta['trained_on_rows']} days")

    fig3, ax3 = plt.subplots(figsize=(8,3))
    merged3 = pd.merge(opd_forecast[["ds","yhat"]], df[["date","opd_visits"]], left_on="ds", right_on="date", how="left")
    ax3.plot(merged3["ds"], merged3["yhat"], label="Predicted")
    opd_lower, opd_upper = safe_get_uncertainty(opd_forecast, opd_model, df, 'opd_visits')
    try:
        lower_al3 = pd.Series(opd_lower.values, index=opd_forecast['ds']).reindex(merged3['ds']).values
        upper_al3 = pd.Series(opd_upper.values, index=opd_forecast['ds']).reindex(merged3['ds']).values
        ax3.fill_between(merged3["ds"], lower_al3, upper_al3, alpha=0.08)
    except Exception:
        ax3.fill_between(merged3["ds"], merged3["yhat"]*0.9, merged3["yhat"]*1.1, alpha=0.06)
    ax3.scatter(merged3["ds"], merged3["opd_visits"], s=8, label="Actual")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("OPD visits")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

    opd_upcoming = opd_forecast[["ds","yhat"]].tail(horizon_days)
    latest_opd = int(df["opd_visits"].iloc[-1]) if "opd_visits" in df.columns else 0
    future_opd = opd_forecast[opd_forecast['ds'] > df['date'].max()]
    next_day_opd = int(round(float(future_opd.iloc[0]["yhat"]))) if not future_opd.empty else int(round(float(opd_upcoming.iloc[0]["yhat"])))
    opd_extra = "OPD: weekday pattern and local events influence visits."
    opd_actions = [
        "Adjust reception staffing (especially on peak days).",
        "If OPD drop & ER increases: triage review."
    ]
    plot_summary_card_html("OPD", latest_opd, next_day_opd, opd_upcoming, extra_text=opd_extra, actions=opd_actions, font_px=22)

    # ---------------- INSIGHTS & SUMMARY ----------------
    trend_er = full_forecast["yhat"].tail(7).mean() - full_forecast["yhat"].tail(14).head(7).mean()
    trend_icu = icu_forecast["yhat"].tail(7).mean() - icu_forecast["yhat"].tail(14).head(7).mean()
    trend_opd = opd_forecast["yhat"].tail(7).mean() - opd_forecast["yhat"].tail(14).head(7).mean()

    try:
        pred_on_history = model.predict(df.rename(columns={'date':'ds'}))
        df['yhat'] = pred_on_history['yhat'].values
        df['resid'] = df['er_visits'] - df['yhat']
        resid_last14 = df['resid'].tail(14)
        resid_mean = resid_last14.mean()
        resid_std = resid_last14.std() if resid_last14.std() > 0 else 1.0
        latest_resid = resid_last14.iloc[-1]
        z = (latest_resid - resid_mean) / resid_std
    except Exception:
        z = 0.0

    inv_df = generate_inventory_recommendations(df)
    staff_df = generate_staffing_recommendations(full_forecast, df, capacity_multiplier)

    sev_counts = {
        "Severe Surge (Red)": len(staff_df[staff_df["overload_percent"] > 15]) if not staff_df.empty else 0,
        "Moderate (Orange)": len(staff_df[(staff_df["overload_percent"] <= 15) & (staff_df["overload_percent"] > 5)]) if not staff_df.empty else 0,
        "Near Capacity (Yellow)": len(staff_df[(staff_df["overload_percent"] <= 5) & (staff_df["overload_percent"] > -5)]) if not staff_df.empty else 0,
        "Under Capacity (Green)": len(staff_df[staff_df["overload_percent"] <= -5]) if not staff_df.empty else 0
    }
    sev = sev_counts["Severe Surge (Red)"]
    mod = sev_counts["Moderate (Orange)"]
    yel = sev_counts["Near Capacity (Yellow)"]

    st.subheader("🧠 Executive Summary")
    if "item" in inv_df.columns and "Oxygen Cylinders" in inv_df["item"].values:
        inv_msg = "oxygen cylinders are running low"
    elif "item" in inv_df.columns and "N95 Masks" in inv_df["item"].values:
        inv_msg = "N95 mask stock is low"
    elif "item" in inv_df.columns and "Paracetamol Strips" in inv_df["item"].values:
        inv_msg = "paracetamol supplies are below threshold"
    else:
        inv_msg = "All inventory levels are healthy."

    anomaly_msg = f"ER anomaly detected (z={z:.2f})" if abs(z) > 2 else "No ER anomalies detected."

    summary_md = textwrap.dedent(f"""
    Over the next week:
    - ER demand change: **{trend_er:+.2f}**
    - ICU demand change: **{trend_icu:+.2f}**
    - OPD demand change: **{trend_opd:+.2f}**

    Surge forecast:
    - **{sev} Severe days**, **{mod} Moderate days**, **{yel} Near-capacity days**

    Inventory: {inv_msg}
    Anomaly Status: {anomaly_msg}
    """).strip()

    st.markdown(f'<div style="font-size:20px; line-height:1.45;">{summary_md.replace("<","&lt;").replace(">","&gt;").replace("\n","  \n")}</div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li {
      font-size:18px !important;
    }
    div[data-testid="stDataFrame"] { font-size:16px !important; }
    div[data-testid="metric-container"] { font-size:20px !important; }
    button[title="Download"] { font-size:16px !important; }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("🧑‍⚕️ Staffing Recommendations (Next 14 Days)")
    if not staff_df.empty:
        st.dataframe(staff_df, use_container_width=True, height=640)
    else:
        st.info("No staffing recommendations generated (check forecast).")

    st.subheader("🔥 Surge Severity Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Severe Surge Days", sev)
    c2.metric("🟠 Moderate Days", mod)
    c3.metric("🟡 Near Capacity", yel)
    c4.metric("🟢 Under Capacity", sev_counts["Under Capacity (Green)"])

    st.subheader("📦 Inventory Recommendations")
    st.dataframe(inv_df, use_container_width=True, height=480)

    st.subheader("📣 Recent Alerts (from Agent Pipeline)")
    if os.path.exists("alerts_log.json"):
        with open("alerts_log.json", "r", encoding="utf-8") as f:
            alerts = json.load(f)
        if alerts:
            st.dataframe(pd.DataFrame(alerts[-20:]), use_container_width=True, height=480)
        else:
            st.info("No alerts yet.")
    else:
        st.info("alerts_log.json not found. Run agent_pipeline.py once.")

    if show_raw:
        st.subheader("📄 Raw Data (Last 30 Days)")
        st.dataframe(df.tail(30), use_container_width=True, height=480)

    st.subheader("📄 Download Daily Report")
    report_html = f"""
    <html><body>
    <h1>SwasthyaAI Daily Report</h1>
    <pre>{summary_md.replace("<", "&lt;").replace(">", "&gt;")}</pre>
    <h2>Inventory</h2>
    {inv_df.to_html(index=False)}
    <h2>Staffing</h2>
    {staff_df.to_html(index=False)}
    </body></html>
    """
    st.download_button("📥 Download Daily Report (HTML)",
                       report_html,
                       "SwasthyaAI_Daily_Report.html",
                       "text/html")

if __name__ == "__main__":
    main()
