import pandas as pd

CAPACITY_MULTIPLIER = 10  
# Meaning: 1 ER staff = capacity to handle 10 patients/day
# You can adjust to 8 or 12 depending on your assumptions


def generate_staffing_recommendations(forecast, original_df):
    """
    forecast: Prophet forecast dataframe
    original_df: the swasthya_ai_data.csv dataframe
    returns: list of staffing recommendation dicts
    """

    # Take only next 14 days of forecast
    forecast_df = forecast[["ds", "yhat"]].tail(14)
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    # Get most recent row (latest staffing)
    latest = original_df.iloc[-1]

    # Convert staff count → actual patient capacity
    er_capacity_staff_count = latest["er_staff_capacity"]
    er_capacity = er_capacity_staff_count * CAPACITY_MULTIPLIER

    staff_actions = []

    print("\n=== STAFFING RECOMMENDATIONS FOR NEXT 14 DAYS ===\n")
    print(f"Computed ER Capacity = {er_capacity_staff_count} staff × {CAPACITY_MULTIPLIER} = {er_capacity} patients/day\n")

    # Loop through the 14 forecasted days
    for i, row in forecast_df.iterrows():
        predicted = row["yhat"]
        date = row["ds"].date()

        # Calculate overload
        overload = (predicted - er_capacity) / er_capacity * 100

        # Rule-based decision engine
        if overload < -5:
            action = "No action needed"
        elif overload < 5:
            action = "Normal day: Monitor staffing"
        elif overload < 15:
            action = "Recommend: Add 1 extra nurse to ER shift"
        elif overload < 25:
            action = "Recommend: Add 1 nurse + 1 medical officer"
        else:
            action = "CRITICAL SURGE: Activate ER surge staffing protocol"

        entry = {
            "date": str(date),
            "predicted_er_visits": round(predicted),
            "er_capacity_patients_per_day": er_capacity,
            "overload_percent": round(overload, 2),
            "recommended_action": action
        }

        staff_actions.append(entry)

        print(
            f"{entry['date']} → Predicted: {entry['predicted_er_visits']} visits "
            f"(Capacity: {er_capacity}, Overload: {entry['overload_percent']}%)\n"
            f"Action: {entry['recommended_action']}\n"
        )

    return staff_actions


# ---------------------------
# Main Script
# ---------------------------
if __name__ == "__main__":

    # Load original dataset
    df = pd.read_csv("swasthya_ai_data.csv")

    # Run Prophet model
    try:
        from prophet import Prophet

        print("Running Prophet model to produce forecast...\n")

        df_prophet = df.rename(columns={"date": "ds", "er_visits": "y"})
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

        # Include regressors
        m = Prophet()
        m.add_regressor("aqi")
        m.add_regressor("temp_c")
        m.add_regressor("festival")

        # Fit model
        m.fit(df_prophet)

        # Predict next 14 days
        future = m.make_future_dataframe(periods=14)
        future["aqi"] = df_prophet["aqi"].iloc[-1]
        future["temp_c"] = df_prophet["temp_c"].iloc[-1]
        future["festival"] = 0

        forecast = m.predict(future)

        # Generate staffing decisions
        generate_staffing_recommendations(forecast, df)

    except Exception as e:
        print("Error occurred while running Prophet:")
        print(e)
