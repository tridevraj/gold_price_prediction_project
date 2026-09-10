"""
generate_data.py
-----------------
Generates a realistic synthetic daily dataset (2015-today) linking gold price
to common macro-economic / financial drivers:
    - USD_Index        : US Dollar Index (gold is usually INVERSELY related)
    - Crude_Oil_Price   : WTI crude oil (mild positive relation - inflation proxy)
    - SP500             : S&P 500 index (risk-on/off proxy, mild negative relation)
    - Interest_Rate     : US Fed funds rate (inverse relation - opportunity cost of holding gold)
    - Inflation_Rate    : CPI YoY % (positive relation - gold is an inflation hedge)
    - Silver_Price       : Silver (positive relation - both precious metals, shares macro drivers)
    - VIX                : Volatility Index (positive relation - safe haven demand)
    - USD_INR            : USD/INR exchange rate (used to convert gold to Rs. per 10g)

Design notes:
    - Gold_Price (USD per troy ounce) is built from a weighted combination of
      the macro drivers PLUS a substantial idiosyncratic component (its own
      random-walk momentum and observation noise) that none of the features
      can explain. This idiosyncratic share is what keeps the downstream
      Random Forest model from achieving an unrealistic ~99% R2 -- real
      gold-price models typically explain roughly 75-85% of price variance
      from macro fundamentals alone.
    - Gold_Price_INR converts this to Rs. per 10 grams -- the standard way
      gold is quoted in India (1 troy ounce = 31.1035 grams).
    - Silver_Price is generated independently from shared macro exposures
      (not derived directly from Gold_Price), so its correlation with gold
      is realistic (~0.4-0.85) instead of near-perfect by construction.
"""

import numpy as np
import pandas as pd
from datetime import date

np.random.seed(42)

# ---------------------------------------------------------------
# 1. Date range: 2015-01-01 through TODAY (so the dataset always has a
#    "current price" row to forecast forward from)
# ---------------------------------------------------------------
dates = pd.bdate_range(start="2015-01-01", end=date.today())
n = len(dates)
t = np.arange(n)

# ---------------------------------------------------------------
# 2. Build correlated macro features using random walks + cycles
# ---------------------------------------------------------------
def random_walk(start, n, drift, vol, seed_offset=0):
    rng = np.random.default_rng(42 + seed_offset)
    steps = rng.normal(drift, vol, n)
    return start + np.cumsum(steps)

usd_index = 95 + 8 * np.sin(t / 500) + random_walk(0, n, 0.0, 0.15, 1) * 0.4
usd_index = np.clip(usd_index, 85, 112)

crude_oil = 60 + 20 * np.sin(t / 400 + 1) + random_walk(0, n, 0.0, 0.4, 2) * 0.5
crude_oil = np.clip(crude_oil, 20, 120)

sp500 = 2000 + t * 1.55 + random_walk(0, n, 0.0, 8, 3)
sp500 = np.clip(sp500, 1800, 6200)

interest_rate = np.piecewise(
    t.astype(float),
    [t < 1200, (t >= 1200) & (t < 1900), (t >= 1900) & (t < 2350), t >= 2350],
    [0.5, lambda x: 0.5 + (x - 1200) * 0.0025, 5.0, lambda x: 5.0 - (x - 2350) * 0.001],
)
interest_rate = np.clip(interest_rate + np.random.normal(0, 0.03, n), 0.05, 5.5)

inflation_rate = 2.0 + 1.2 * np.sin(t / 600) + np.random.normal(0, 0.15, n)
covid_infl_boost = np.where((t > 1550) & (t < 2000), (t - 1550) / 450 * 6.5, 0)
inflation_rate = np.clip(inflation_rate + covid_infl_boost, 0.0, 9.5)

vix = 15 + 5 * np.abs(np.sin(t / 250)) + np.random.normal(0, 2, n)
crisis_days = np.random.choice(n, size=25, replace=False)
for d in crisis_days:
    spread = np.exp(-np.abs(np.arange(n) - d) / 8)
    vix = vix + spread * np.random.uniform(15, 45)
vix = np.clip(vix, 9, 85)

usd_inr = 63 + t * 0.0095 + random_walk(0, n, 0.0, 0.15, 4)
usd_inr = np.clip(usd_inr, 62, 88)

# ---------------------------------------------------------------
# 3. Construct Gold Price: macro signal + large idiosyncratic component
# ---------------------------------------------------------------
def z(x):
    return (x - x.mean()) / x.std()

latent_trend = 1100 + t * 0.62  # long-run secular uptrend in gold

gold_macro_signal = (
    latent_trend
    - 42 * z(usd_index)          # inverse relation
    + 14 * z(crude_oil)          # mild positive (inflation-linked)
    - 9 * z(sp500)               # mild inverse (risk-on reduces safe-haven demand)
    - 22 * z(interest_rate)      # inverse (opportunity cost)
    + 30 * z(inflation_rate)     # positive (inflation hedge)
    + 18 * z(vix)                # positive (safe haven in volatility)
    + 11 * z(usd_inr)            # positive (weaker USD relative EM currencies)
)

# Idiosyncratic gold dynamics: own momentum (random walk) + observation noise.
# This is intentionally sized so the macro features explain roughly ~80% of
# total price variance -- tuned empirically against the modeling pipeline.
own_momentum = np.cumsum(np.random.normal(0, 3.2, n)) * 4.6
obs_noise = np.random.normal(0, 370, n)

gold_price = gold_macro_signal + own_momentum + obs_noise
gold_price = np.clip(gold_price, 800, 4500)

# ---------------------------------------------------------------
# 3b. Convert to INR per 10 grams (the standard way gold is quoted in India)
#     1 troy ounce = 31.1035 grams
# ---------------------------------------------------------------
TROY_OUNCE_TO_GRAMS = 31.1035
gold_price_inr = (gold_price / TROY_OUNCE_TO_GRAMS) * 10 * usd_inr

# ---------------------------------------------------------------
# 4. Silver: shares SOME of gold's macro exposure, but built independently
#    (not derived from the realized Gold_Price) -> realistic ~0.4-0.85 corr
# ---------------------------------------------------------------
silver_macro_signal = (
    16
    - 1.1 * z(usd_index)
    + 0.6 * z(crude_oil)
    + 1.3 * z(inflation_rate)
    + 0.5 * z(vix)
    + 0.03 * (t / n) * 10          # mild long-run uptrend, own scale
)
silver_own_noise = np.cumsum(np.random.normal(0, 0.05, n)) + np.random.normal(0, 0.9, n)
silver_price = silver_macro_signal + silver_own_noise
silver_price = np.clip(silver_price, 11, 34)

# ---------------------------------------------------------------
# 5. Assemble DataFrame
# ---------------------------------------------------------------
df = pd.DataFrame({
    "Date": dates,
    "Gold_Price": gold_price.round(2),
    "Gold_Price_INR": gold_price_inr.round(2),
    "Silver_Price": silver_price.round(2),
    "USD_Index": usd_index.round(2),
    "Crude_Oil_Price": crude_oil.round(2),
    "SP500": sp500.round(2),
    "Interest_Rate": interest_rate.round(3),
    "Inflation_Rate": inflation_rate.round(2),
    "VIX": vix.round(2),
    "USD_INR": usd_inr.round(2),
})

if __name__ == "__main__":
    out_path = "data/gold_price_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.head())
    print(df.describe().T[["mean", "std", "min", "max"]])
    print("\nCorrelation of Gold_Price with Silver_Price:",
          df["Gold_Price"].corr(df["Silver_Price"]))
    print("\nSample Gold_Price_INR (Rs. per 10g):", df["Gold_Price_INR"].iloc[-1])