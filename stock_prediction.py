# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from viz_utils import plot_candles, plot_moving_window_boxplots

# NEW: our helper for robust loading
from data_utils import load_and_process_data

# >>> NEW (Task C.4): model builder & trainer with early stopping/checkpoints
from model_utils import build_sequence_model, train_and_evaluate
# <<< NEW

#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------
# DATA_SOURCE = "yahoo"
COMPANY = 'CBA.AX'

TRAIN_START = '2018-01-01'     # Start date to read
TRAIN_END = '2023-08-01'       # End date to read

# data = web.DataReader(COMPANY, DATA_SOURCE, TRAIN_START, TRAIN_END) # Read data using yahoo

import yfinance as yf

# Get the data for the stock AAPL
# (kept here to preserve original comments; not used by the new loader)
data = yf.download(COMPANY, TRAIN_START, TRAIN_END)

#------------------------------------------------------------------------------
# Prepare Data
## To do:
# 1) Check if data has been prepared before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Use a different price value eg. mid-point of Open & Close
# 3) Change the Prediction days
#------------------------------------------------------------------------------
# === New, robust multi-feature loader (replaces the old single-column code) ===
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
PREDICTION_DAYS = 30      # window length (a.k.a. n_steps)
LOOKUP_STEP = 1           # next-day horizon

bundle = load_and_process_data(
    ticker=COMPANY,
    start_date=TRAIN_START,
    end_date=TRAIN_END,
    feature_columns=FEATURE_COLUMNS,
    n_steps=PREDICTION_DAYS,
    lookup_step=LOOKUP_STEP,
    scale=True,
    split_by_date=True,       # set False for random split
    test_size=0.2,
    shuffle=True,
    nan_mode="ffill_bfill",
    cache_dir="cache",        # set None to disable caching
    force_refresh=False,
    auto_adjust=True,         # Close is already adjusted; 'Adj Close' may be absent
)

# Use arrays v0.1 expects
x_train, y_train = bundle.X_train, bundle.y_train
x_test,  y_test  = bundle.X_test,  bundle.y_test

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
# >>> UPDATED (Task C.4): build via model factory and train with callbacks
# IMPORTANT CHANGE: input is now (PREDICTION_DAYS, number_of_features)
model = build_sequence_model(
    input_shape=(x_train.shape[1], x_train.shape[2]), #(timesteps, features)
    layers=[
        {"type": "LSTM", "units": 32, "return_sequences": False, "dropout": 0.1},
    ],
    dense_units=1,
    optimizer="adam",
    loss="mean_squared_error",
)


# Train with early-stopping and best-weight checkpoint (saved in results/)
_ = train_and_evaluate(
    model,
    bundle,
    epochs=10,
    batch_size=128,
    patience=2,                 # stop if val_loss stalls
    results_dir="results",
    run_name="taskc4_lstm_2x50",
    plot_pred=False,            
)
# <<< UPDATED

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# We already created x_test / y_test in the loader, so we can predict directly.
predicted_scaled = model.predict(x_test)

# Inverse-scale predicted and true values using the 'adjclose' scaler if we scaled
if "adjclose" in bundle.column_scaler:
    scaler = bundle.column_scaler["adjclose"]
    predicted_prices = scaler.inverse_transform(predicted_scaled)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
else:
    predicted_prices = predicted_scaled
    actual_prices = y_test

actual = actual_prices.astype(float)
pred   = predicted_prices.ravel().astype(float)

mae  = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
mape = np.mean(np.abs((actual - pred) / actual)) * 100

print(f"Test MAE:  {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAPE: {mape:.2f}%")

#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------
plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------
# Use the last sequence prepared by the loader to predict the next price
last_seq = bundle.last_sequence[-bundle.n_steps:]  # last n_steps rows
last_seq = np.expand_dims(last_seq, axis=0)        # (1, n_steps, n_features)
next_scaled = model.predict(last_seq)

if "adjclose" in bundle.column_scaler:
    next_price = bundle.column_scaler["adjclose"].inverse_transform(next_scaled)[0, 0]
else:
    next_price = float(next_scaled[0, 0])

print(f"Prediction: {next_price}")

# 1) Candlesticks with n-day candles (e.g., weekly-like 5 trading days per candle)
# Use raw, unscaled prices for viz (bundle.df has standardized raw columns)
df_for_viz = bundle.df.copy()

plot_candles(
    df_for_viz,
    n=5,                    # 5 trading sessions per candle
    ma_window=20,           # blue SMA(20) + legend
    title=f"{COMPANY} – 5-day Candlesticks",
    volume=True,
    hover_ma=True           # tooltip on the blue line
)

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??
