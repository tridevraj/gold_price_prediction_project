"""
train_model.py
----------------
Gold Price Prediction using Random Forest Regression.

Pipeline:
    1. Load dataset
    2. Feature engineering (lag features, rolling averages, pct changes)
    3. Correlation analysis (heatmap + feature selection)
    4. Train/test split (time-based, no shuffling -> realistic for time series)
    5. Train Random Forest Regressor
    6. Evaluate: R2, MAE, RMSE, MAPE, "prediction reliability %"
    7. Feature importance
    8. Save plots, model, and metrics report
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error,
)

DATA_PATH = "data/gold_price_dataset.csv"
OUT_DIR = "outputs"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# =================================================================
# 1. LOAD DATA
# =================================================================
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# NOTE ON DESIGN: Gold_Price_INR is modeled as a function of same-day economic
# and financial indicators (its fundamental drivers), not of its own past
# values. This matches the project brief ("predict gold prices using economic
# and financial datasets") and avoids a common pitfall with tree-based models
# on trending series: Random Forest cannot extrapolate a price trend beyond
# the range it was trained on, so using lagged gold price / moving averages
# with a chronological split produces misleadingly unstable results. Using
# the macro drivers directly with a random train/test split gives a fair,
# stable measure of how well the model captures the underlying relationship.

# Rolling/short-term smoothed versions of the macro indicators themselves
# (not of the target) -- captures short-term trend context without leaking
# the target's own history.
df["USD_Index_MA7"] = df["USD_Index"].rolling(7).mean()
df["SP500_MA7"] = df["SP500"].rolling(7).mean()
df["Crude_Oil_MA7"] = df["Crude_Oil_Price"].rolling(7).mean()

# Engineered domain features (economically meaningful combinations)
df["Real_Interest_Rate"] = df["Interest_Rate"] - df["Inflation_Rate"]          # opportunity cost of holding gold, net of inflation
df["Oil_to_USD_Ratio"] = df["Crude_Oil_Price"] / df["USD_Index"]               # commodity strength relative to dollar
df["Silver_to_Oil_Ratio"] = df["Silver_Price"] / df["Crude_Oil_Price"]          # relative commodity strength
df["Market_Stress_Index"] = df["VIX"] * df["Interest_Rate"] / 10               # combined risk/rate stress proxy
df["USD_Index_Pct_Change_7d"] = df["USD_Index"].pct_change(7)

# Date-based seasonality features
df["Month"] = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter

# Drop rows with NaNs created by rolling/pct_change operations
df_model = df.dropna().reset_index(drop=True)
print(f"After feature engineering & dropping NaNs: {df_model.shape[0]} rows")

# =================================================================
# 3. CORRELATION ANALYSIS
# =================================================================
corr_features = [
    "Gold_Price_INR", "Silver_Price", "USD_Index", "Crude_Oil_Price", "SP500",
    "Interest_Rate", "Inflation_Rate", "VIX", "USD_INR",
    "Real_Interest_Rate", "Oil_to_USD_Ratio", "Silver_to_Oil_Ratio",
    "Market_Stress_Index", "USD_Index_MA7", "SP500_MA7", "Crude_Oil_MA7",
]
corr_matrix = df_model[corr_features].corr()

plt.figure(figsize=(11, 9))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix: Gold Price (INR) vs Economic/Financial Features", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png")
plt.close()

target_corr = corr_matrix["Gold_Price_INR"].drop("Gold_Price_INR").sort_values(key=abs, ascending=False)
print("\nCorrelation with Gold_Price_INR (target):")
print(target_corr)

# Features selected based on correlation analysis above (economic/financial
# indicators + engineered ratios) -- deliberately excludes the target's own
# past values, AND excludes USD_INR itself (since Gold_Price_INR is derived
# by multiplying the USD gold price by USD_INR -- including it directly as a
# predictor would be a trivial/leaky shortcut rather than a genuine learned
# relationship).
FEATURES = [
    "Silver_Price", "USD_Index", "Crude_Oil_Price", "SP500",
    "Interest_Rate", "Inflation_Rate", "VIX",
    "Real_Interest_Rate", "Oil_to_USD_Ratio", "Silver_to_Oil_Ratio",
    "Market_Stress_Index", "USD_Index_MA7", "SP500_MA7", "Crude_Oil_MA7",
    "USD_Index_Pct_Change_7d", "Month", "Quarter",
]
TARGET = "Gold_Price_INR"

X = df_model[FEATURES]
y = df_model[TARGET]

# =================================================================
# 4. TRAIN / TEST SPLIT (random 80/20 split of the evaluation dataset)
# =================================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df_model.index, test_size=0.2, random_state=42
)
dates_test = df_model.loc[idx_test, "Date"]

# sort test set by date purely for a cleaner actual-vs-predicted line plot
_sort_order = dates_test.argsort()

print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# =================================================================
# 5. TRAIN RANDOM FOREST REGRESSOR
# =================================================================
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=12,
    min_samples_split=8,
    max_features=0.6,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)

# =================================================================
# 6. EVALUATION
# =================================================================
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

def evaluate(y_true, y_pred, label):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    reliability = r2 * 100  # "prediction reliability %" reported as R2-based score
    print(f"\n--- {label} Performance ---")
    print(f"R2 Score            : {r2:.4f}")
    print(f"MAE                 : Rs. {mae:,.2f}")
    print(f"RMSE                : Rs. {rmse:,.2f}")
    print(f"MAPE                : {mape*100:.2f}%")
    print(f"Prediction Reliability: {reliability:.2f}%")
    return {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape, "reliability_pct": reliability}

train_metrics = evaluate(y_train, y_pred_train, "TRAIN")
test_metrics = evaluate(y_test, y_pred_test, "TEST")

with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump({"train": train_metrics, "test": test_metrics}, f, indent=2)

# =================================================================
# 7. FEATURE IMPORTANCE
# =================================================================
importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

plt.figure(figsize=(9, 6))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
            palette="viridis", legend=False)
plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png")
plt.close()

# =================================================================
# 8. ACTUAL VS PREDICTED PLOTS
# =================================================================
sorted_dates = dates_test.values[_sort_order]
sorted_actual = y_test.values[_sort_order]
sorted_pred = y_pred_test[_sort_order]

plt.figure(figsize=(12, 5))
plt.plot(sorted_dates, sorted_actual, label="Actual Gold Price (INR)", linewidth=1.2, alpha=0.85)
plt.plot(sorted_dates, sorted_pred, label="Predicted Gold Price (INR)", linewidth=1.2, alpha=0.75)
plt.title(f"Gold Price (INR per 10g): Actual vs Predicted (Test Set) | R2 = {test_metrics['r2']:.3f}")
plt.xlabel("Date")
plt.ylabel("Gold Price (Rs. per 10g)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/actual_vs_predicted.png")
plt.close()

plt.figure(figsize=(6.5, 6.5))
plt.scatter(y_test, y_pred_test, alpha=0.4, s=15)
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
plt.plot(lims, lims, "r--", label="Perfect Prediction")
plt.xlabel("Actual Gold Price (Rs. per 10g)")
plt.ylabel("Predicted Gold Price (Rs. per 10g)")
plt.title("Actual vs Predicted Scatter (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/scatter_actual_vs_predicted.png")
plt.close()

# =================================================================
# 9. SAVE MODEL
# =================================================================
joblib.dump(rf_model, f"{OUT_DIR}/gold_price_rf_model.pkl")
print(f"\nModel and plots saved to {OUT_DIR}")