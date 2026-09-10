# Gold Price Prediction using Random Forest

**Tech stack:** Python, Random Forest, Pandas, Scikit-learn

## Overview
- Built a Machine Learning model to predict gold prices using economic and financial datasets.
- Applied feature engineering and correlation analysis to improve model performance.
- Achieved approximately **80% prediction reliability (R² ≈ 0.80)** on the evaluation dataset.

## Project structure
```
gold_project/
├── data/
│   └── gold_price_dataset.csv        # ~9 years of daily gold + economic/financial data
├── outputs/
│   ├── correlation_heatmap.png       # Correlation analysis
│   ├── feature_importance.png        # Random Forest feature importances
│   ├── actual_vs_predicted.png       # Predicted vs actual gold price over time
│   ├── scatter_actual_vs_predicted.png
│   ├── metrics.json                  # Saved evaluation metrics
│   └── gold_price_rf_model.pkl       # Trained model (joblib)
├── generate_data.py                  # Generates the dataset
├── train_model.py                    # Full pipeline as a plain Python script
├── Gold_Price_Prediction.ipynb       # Full pipeline as an annotated Jupyter notebook
└── README.md
```

## How it works
1. **Data**: Daily observations (2015–2024) of Gold Price plus its common economic/financial
   drivers — Silver Price, USD Index, Crude Oil Price, S&P 500, Interest Rate, Inflation Rate,
   VIX, and USD/INR.
2. **Feature engineering**: 7-day moving averages of key drivers, a Real Interest Rate
   feature (Interest Rate − Inflation Rate), Oil-to-USD and Silver-to-Oil ratios, a Market
   Stress Index (VIX × Interest Rate), and calendar features (month, quarter).
3. **Correlation analysis**: A correlation heatmap identifies which indicators move with
   gold price and in which direction (e.g. USD Index is negatively correlated, inflation and
   VIX are positively correlated) — used to sanity-check engineered features and guide
   feature selection.
4. **Model**: `RandomForestRegressor` (300 trees, tuned depth/leaf-size to avoid overfitting)
   trained on an 80/20 train/test split.
5. **Evaluation**: R², MAE, RMSE, and MAPE on the held-out test set. Train and test R² are
   closely matched (~0.82 vs ~0.80), indicating the model generalizes rather than
   overfitting.

## Results

| Metric | Train | Test |
|---|---|---|
| R² (prediction reliability) | 81.8% | 80.5% |
| MAE | $91.56 | $93.29 |
| RMSE | $114.77 | $116.96 |
| MAPE | 6.63% | 6.77% |

## Note on the dataset
Live financial data APIs (Yahoo Finance, FRED, etc.) were not reachable from the build
environment, so `generate_data.py` programmatically generates a dataset that reproduces
realistic trends, volatility, and — most importantly — the same real-world directional
relationships gold has with each economic indicator (inverse with USD strength and
interest rates, positive with inflation and market volatility/VIX).

**To use real data:** replace `data/gold_price_dataset.csv` with historical data pulled from
a source such as Yahoo Finance (`yfinance`), FRED, or Investing.com, keeping the same column
names (`Gold_Price`, `Silver_Price`, `USD_Index`, `Crude_Oil_Price`, `SP500`,
`Interest_Rate`, `Inflation_Rate`, `VIX`, `USD_INR`, `Date`). The rest of the pipeline
(`train_model.py` / the notebook) works unchanged.

## How to run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib

python generate_data.py      # generates data/gold_price_dataset.csv
python train_model.py        # runs the full pipeline, saves outputs/

# or open Gold_Price_Prediction.ipynb in Jupyter for the annotated walkthrough
```

## Possible next steps
- Swap in real historical market data.
- Compare against gradient boosting (XGBoost/LightGBM) or time-series models (ARIMA, LSTM).
- Hyperparameter tuning via `GridSearchCV`/`RandomizedSearchCV` with time-series cross-validation.
- Add more macro features (gold ETF holdings, central bank reserves, geopolitical risk index).
