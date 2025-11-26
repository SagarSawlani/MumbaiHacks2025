import pandas as pd

def generate_inventory_recommendations(original_df):
    """
    original_df: the swasthya_ai_data.csv dataframe
    returns: list of inventory recommendation dicts
    """

    print("\n=== INVENTORY RECOMMENDATIONS ===\n")

    # Get last 7 days usage averages
    oxygen_daily_usage = original_df["oxygen_used"].tail(7).mean()
    n95_daily_usage = original_df["n95_used"].tail(7).mean()
    para_daily_usage = original_df["para_used"].tail(7).mean()

    # Projected usage for next 7 days
    projected_o2_usage = oxygen_daily_usage * 7
    projected_n95_usage = n95_daily_usage * 7
    projected_para_usage = para_daily_usage * 7

    # Get current stock levels (last row)
    latest = original_df.iloc[-1]

    o2_stock = latest["oxygen_stock"]
    n95_stock = latest["n95_stock"]
    para_stock = latest["para_stock"]

    actions = []

    # -------- RULES ----------
    # Oxygen: Minimum buffer = 20 cylinders
    if o2_stock - projected_o2_usage < 20:
        actions.append({
            "item": "Oxygen Cylinders",
            "current_stock": o2_stock,
            "projected_7day_usage": round(projected_o2_usage),
            "recommendation": "Order 20 oxygen cylinders immediately"
        })

    # N95 masks: Minimum safe threshold = 200
    if n95_stock < 200:
        actions.append({
            "item": "N95 Masks",
            "current_stock": n95_stock,
            "projected_7day_usage": round(projected_n95_usage),
            "recommendation": "Order 200 N95 masks"
        })

    # Paracetamol: Minimum safe threshold = 300 strips
    if para_stock < 300:
        actions.append({
            "item": "Paracetamol Strips",
            "current_stock": para_stock,
            "projected_7day_usage": round(projected_para_usage),
            "recommendation": "Order 300 paracetamol strips"
        })

    # -------- OUTPUT ----------
    if not actions:
        print("Inventory levels are healthy. No orders needed.")
    else:
        for a in actions:
            print(
                f"Item: {a['item']}\n"
                f"  Current Stock: {a['current_stock']}\n"
                f"  Projected 7-day usage: {a['projected_7day_usage']}\n"
                f"  Recommendation: {a['recommendation']}\n"
            )

    return actions



# ---------------------------
# Example usage:
# ---------------------------
if __name__ == "__main__":

    df = pd.read_csv("swasthya_ai_data.csv")

    generate_inventory_recommendations(df)
