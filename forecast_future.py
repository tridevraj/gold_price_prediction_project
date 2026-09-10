"""
forecast_future.py
--------------------
Extends the project to forecast gold price FORWARD in time (not just explain
today's price from today's conditions).

Key idea: Random Forest cannot extrapolate an absolute price level beyond the
range it saw during training (trees just predict averages of training leaves).
So instead of predicting the price level directly, this script predicts the
DAY-OVER-DAY CHANGE in gold price (a roughly stationary quantity, not
trending), using:
    - Lagged price changes (recent momentum: lag 1, 2, 3, 5, 10 days)
    - Rolling mean/std of recent price changes (trend strength & volatility)
    - Lagged macro/financial indicators (yesterday's USD Index, Interest Rate,
      Inflation, VIX, etc. -- using LAGGED values only, since in a real
      forecasting setting you only know past values, not future ones)

It then reconstructs the forecasted price by adding predicted changes onto the
last known price, one step at a time (iterative / recursive forecasting), for
a chosen number of business days into the future.

IMPORTANT CAVEAT (shown in the output too): future values of the macro
indicators themselves (USD Index, Interest Rate, etc.) are NOT known in
advance. This script holds them at their most recently observed values for
the forecast horizon, which is a simplifying assumption. For more realistic
forecasts, replace `future_macro_assumption()` with actual forecasts/expected
values for those indicators (e.g. from economic forecasts or your own
scenario assumptions).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_PATH = "data/gold_price_dataset.csv"
OUT_DIR = "outputs"
FORECAST_DAYS = 10  # how many business days ahead to forecast

# =================================================================
# 1. LOAD DATA
# =================================================================
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# =================================================================
# 2. FEATURE ENGINEERING FOR FORECASTING
#    (predict the CHANGE in price, using only information available
#    at the time -- i.e. everything is lagged / historical)
# =================================================================
df["Gold_Price_Diff"] = df["Gold_Price_INR"].diff()

for lag in [1, 2, 3, 5, 10]:
    df[f"Diff_Lag{lag}"] = df["Gold_Price_Diff"].shift(lag)

df["Diff_RollMean5"] = df["Gold_Price_Diff"].shift(1).rolling(5).mean()
df["Diff_RollMean10"] = df["Gold_Price_Diff"].shift(1).rolling(10).mean()
df["Diff_RollStd10"] = df["Gold_Price_Diff"].shift(1).rolling(10).std()

macro_cols = ["Silver_Price", "USD_Index", "Crude_Oil_Price", "SP500",
              "Interest_Rate", "Inflation_Rate", "VIX"]
for col in macro_cols:
    df[f"{col}_Lag1"] = df[col].shift(1)

FEATURES = (
    [f"Diff_Lag{lag}" for lag in [1, 2, 3, 5, 10]]
    + ["Diff_RollMean5", "Diff_RollMean10", "Diff_RollStd10"]
    + [f"{col}_Lag1" for col in macro_cols]
)
TARGET = "Gold_Price_Diff"

df_model = df.dropna().reset_index(drop=True)
print(f"Rows available for forecasting model: {len(df_model)}")

# =================================================================
# 3. TIME-BASED TRAIN/TEST SPLIT (chronological, since this is real
#    forecasting -- test on the most recent period, unseen in training)
# =================================================================
split_idx = int(len(df_model) * 0.85)
train_df = df_model.iloc[:split_idx]
test_df = df_model.iloc[split_idx:]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# =================================================================
# 4. TRAIN THE FORECASTING MODEL (predicts next-day price CHANGE)
# =================================================================
forecast_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=15,
    min_samples_split=10,
    max_features=0.6,
    random_state=42,
    n_jobs=-1,
)
forecast_model.fit(X_train, y_train)

# --- Evaluate one-step-ahead accuracy on the held-out (future) period ---
pred_diff_test = forecast_model.predict(X_test)

# Reconstruct actual price levels from predicted diffs for a fair, readable metric
actual_price = test_df["Gold_Price_INR"].values
prev_price = df_model["Gold_Price_INR"].iloc[split_idx - 1:split_idx + len(test_df) - 1].values
predicted_price = prev_price + pred_diff_test

r2 = r2_score(actual_price, predicted_price)
mae = mean_absolute_error(actual_price, predicted_price)
rmse = np.sqrt(mean_squared_error(actual_price, predicted_price))

print("\n--- One-step-ahead forecast accuracy (on unseen future period) ---")
print(f"R2 Score : {r2:.4f}")
print(f"MAE      : Rs. {mae:,.2f}")
print(f"RMSE     : Rs. {rmse:,.2f}")
print(f"Prediction Reliability: {r2*100:.2f}%")

# =================================================================
# 5. MULTI-STEP FUTURE FORECAST (iterative / recursive)
# =================================================================
def future_macro_assumption(last_row):
    """
    Simplifying assumption: hold each macro indicator constant at its last
    observed value for the forecast horizon. Replace this with real forecasts
    or your own scenario values (e.g. expected Fed rate path) for a more
    realistic outlook.
    """
    return {col: last_row[col] for col in macro_cols}

def forecast_future_prices(df_full, model, n_days=30):
    history = df_full.copy().reset_index(drop=True)
    last_row = history.iloc[-1]
    macro_future = future_macro_assumption(last_row)

    forecasts = []
    current_price = last_row["Gold_Price_INR"]
    diff_history = list(history["Gold_Price_Diff"].dropna().values[-15:])

    last_date = history["Date"].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

    for date in future_dates:
        feat = {}
        for lag in [1, 2, 3, 5, 10]:
            feat[f"Diff_Lag{lag}"] = diff_history[-lag]
        feat["Diff_RollMean5"] = np.mean(diff_history[-5:])
        feat["Diff_RollMean10"] = np.mean(diff_history[-10:])
        feat["Diff_RollStd10"] = np.std(diff_history[-10:])
        for col in macro_cols:
            feat[f"{col}_Lag1"] = macro_future[col]

        X_next = pd.DataFrame([feat])[FEATURES]
        predicted_diff = model.predict(X_next)[0]

        current_price = current_price + predicted_diff
        diff_history.append(predicted_diff)

        forecasts.append({"Date": date, "Forecasted_Gold_Price": round(current_price, 2)})

    return pd.DataFrame(forecasts)

forecast_df = forecast_future_prices(df_model, forecast_model, n_days=FORECAST_DAYS)
forecast_df.to_csv(f"{OUT_DIR}/future_forecast.csv", index=False)

last_10 = df_model[["Date", "Gold_Price_INR"]].tail(10).copy()
last_10["Date"] = last_10["Date"].dt.strftime("%Y-%m-%d")
current_price = df_model["Gold_Price_INR"].iloc[-1]
current_date = df_model["Date"].iloc[-1].strftime("%Y-%m-%d")

print("\n--- Past 10 days' Gold Price ---")
print(last_10.to_string(index=False))

print(f"\n--- Current price ({current_date}) ---")
print(f"Rs. {current_price:,.2f}")

forecast_df_display = forecast_df.copy()
forecast_df_display["Date"] = forecast_df_display["Date"].dt.strftime("%Y-%m-%d")
print(f"\n--- Forecast for the next {FORECAST_DAYS} business days ---")
print(forecast_df_display.to_string(index=False))
print(f"\nSaved forecast to {OUT_DIR}/future_forecast.csv")
print("\nNOTE 1: The dataset used here is SYNTHETIC (generated to realistically")
print("mimic gold price behavior, since live market data wasn't reachable from")
print("this environment) -- so 'current price' above is a realistic demo value,")
print("not the real live gold price. Swap in real historical data for real use.")
print("\nNOTE 2: This forecast assumes macro indicators (USD Index, interest")
print("rates, etc.) stay at their most recently observed values. Real-world")
print("forecasts should replace this assumption with actual projections.")
print("\nNOTE 3: Day-to-day price forecasting is inherently much harder than")
print("explaining price from same-day conditions -- treat this forecast as a")
print("directional estimate, not a precise prediction.")

# =================================================================
# 6. PLOT: recent history + forecast
# =================================================================
recent_history = df_model[["Date", "Gold_Price_INR"]].tail(120)

plt.figure(figsize=(12, 5))
plt.plot(recent_history["Date"], recent_history["Gold_Price_INR"],
         label="Historical Gold Price (INR)", linewidth=1.4, color="darkgoldenrod")
plt.plot(forecast_df["Date"], forecast_df["Forecasted_Gold_Price"],
         label=f"Forecast (next {FORECAST_DAYS} business days)",
         linewidth=1.6, linestyle="--", color="crimson")
plt.axvline(recent_history["Date"].iloc[-1], color="gray", linestyle=":", linewidth=1)
plt.title("Gold Price (Rs. per 10g): Recent History + Future Forecast")
plt.xlabel("Date")
plt.ylabel("Gold Price (Rs. per 10g)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/future_forecast.png")
plt.close()

print(f"Saved forecast chart to {OUT_DIR}/future_forecast.png")
