# Gold Price Analysis and Forecasting

A small Python project with two deliberately separate tasks:

- `train_model.py` estimates same-day gold value from same-day economic indicators. Its chronological holdout tests later dates, but because those indicators are observed on the same date, this is a retrospective relationship model—not a future forecast.
- `forecast_future.py` predicts daily price changes from lagged values and recursively projects a short horizon. Future macro indicators are held constant at their latest values, so treat the output as a scenario illustration.

The bundled dataset is synthetic. Its metrics and forecasts are demonstrations and must not be represented as live market predictions.

## Project files

- `generate_data.py`: makes the reproducible synthetic dataset.
- `train_model.py`: checks and engineers data, creates a chronological 80/20 holdout, trains and evaluates a Random Forest, and saves plots, metrics, and the model under `outputs/`.
- `forecast_future.py`: evaluates a next-day change model on the latest chronological holdout and creates a recursive 10-business-day scenario forecast.
- `Gold_Price_Prediction.ipynb`: concise notebook entry point for inspection and running both scripts.

## Setup and run

```bash
python -m pip install -r requirements.txt
python generate_data.py
python train_model.py
python forecast_future.py
```

Run commands from any directory; data and output paths are resolved relative to each script. Both modeling scripts create `outputs/` as needed and check required input columns and unique dates.

## Interpreting results

R² is a statistical fit score, not a percentage probability that an individual prediction is correct. Compare MAE/RMSE to a simple baseline and evaluate on a later time period. A shuffled split can leak future market regimes into training, so the scripts use chronological holdouts. In the latest run, the same-day model scored R² = -1.51 on the holdout and the change model scored R² = -0.003. The models did not demonstrate useful out-of-time accuracy; do not rely on the forecasts. This weakness reflects the synthetic series’ drift and random shocks, and should be investigated with real, correctly aligned data before use.

The synthetic generator encodes trends and relationships by construction. Good scores on generated data do not demonstrate that a model will work on real markets. For real research, replace the CSV with sourced and aligned observations, document units and publication timing, and ensure every feature was actually available when a forecast would have been made. Macro releases often arrive with delays and revisions; same-day values can therefore leak information even when the date split is chronological.

The INR target is quoted per 10 grams. In the recursive forecast, macro variables remain constant and model-predicted price changes feed subsequent steps, so uncertainty can accumulate over the forecast horizon.
