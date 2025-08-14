import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stock_prediction import create_model, load_data
from parameters import *


def plot_graph(test_df):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


def get_final_df(model, data):
    """
    Build a final dataframe with current, predicted, and true future prices.
    Ensures everything is 1-D numeric so downstream math works.
    """
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Predict (scaled)
    y_pred = model.predict(X_test)

    # Inverse-scale predicted & true values to real prices
    if SCALE:
        y_test_inv = data["column_scaler"]["adjclose"].inverse_transform(
            y_test.reshape(-1, 1)
        ).ravel()
        y_pred_inv = data["column_scaler"]["adjclose"].inverse_transform(
            y_pred.reshape(-1, 1)
        ).ravel()
    else:
        y_test_inv = y_test.ravel()
        y_pred_inv = y_pred.ravel()

    # Start from the test features DataFrame (already REAL units)
    final_df = data["test_df"].copy()

    # --- make columns single-level & lowercase ---
    if isinstance(final_df.columns, pd.MultiIndex):
        final_df.columns = [
            "_".join([str(x) for x in col if x is not None]).strip().lower()
            for col in final_df.columns
        ]
    else:
        final_df.columns = [str(c).strip().lower() for c in final_df.columns]

    # drop duplicate-named columns if any
    final_df = final_df.loc[:, ~final_df.columns.duplicated(keep="first")]

    # find the current-price column (any column containing 'adjclose')
    adj_candidates = [c for c in final_df.columns if "adjclose" in c]
    adj_candidates.sort(key=lambda c: (c != "adjclose", len(c)))  # prefer exact 'adjclose'
    if not adj_candidates:
        raise KeyError(f"No adjclose-like column found. Columns: {list(final_df.columns)}")
    adj_col = adj_candidates[0]

    # current price is ALREADY in real units; do NOT inverse-scale again
    current_series = pd.to_numeric(final_df[adj_col].squeeze(), errors="coerce").to_numpy()
    final_df["current_adjclose"] = current_series.astype(float)

    # attach predicted & true future prices (already inverse-scaled)
    final_df[f"adjclose_{LOOKUP_STEP}"] = y_pred_inv.astype(float)
    final_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test_inv.astype(float)
    final_df.sort_index(inplace=True)

    # profits using clean numeric columns
    buy_profit  = lambda current, pred_future, true_future: (true_future - current) if pred_future > current else 0.0
    sell_profit = lambda current, pred_future, true_future: (current - true_future) if pred_future < current else 0.0

    final_df["buy_profit"] = list(map(
        buy_profit,
        final_df["current_adjclose"].to_numpy(),
        final_df[f"adjclose_{LOOKUP_STEP}"].to_numpy(),
        final_df[f"true_adjclose_{LOOKUP_STEP}"].to_numpy()
    ))
    final_df["sell_profit"] = list(map(
        sell_profit,
        final_df["current_adjclose"].to_numpy(),
        final_df[f"adjclose_{LOOKUP_STEP}"].to_numpy(),
        final_df[f"true_adjclose_{LOOKUP_STEP}"].to_numpy()
    ))

    return final_df






def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price


# load the data
data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                feature_columns=FEATURE_COLUMNS)

# construct the model
model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

# load optimal model weights from results folder
model_path = os.path.join("results", model_name + ".weights.h5")
model.load_weights(model_path)

# evaluate the model
loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
if SCALE:
    mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
else:
    mean_absolute_error = mae

# get the final dataframe for the testing set
final_df = get_final_df(model, data)
# predict the future price
future_price = predict(model, data)
# we calculate the accuracy by counting the number of positive profits
accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
# calculating total buy & sell profit
total_buy_profit  = final_df["buy_profit"].sum()
total_sell_profit = final_df["sell_profit"].sum()
# total profit by adding sell & buy together
total_profit = total_buy_profit + total_sell_profit
# dividing total profit by number of testing samples (number of trades)
profit_per_trade = total_profit / len(final_df)
# printing metrics
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
print(f"{LOSS} loss:", loss)
print("Mean Absolute Error:", mean_absolute_error)
print("Accuracy score:", accuracy_score)
print("Total buy profit:", total_buy_profit)
print("Total sell profit:", total_sell_profit)
print("Total profit:", total_profit)
print("Profit per trade:", profit_per_trade)
# plot true/pred prices graph
plot_graph(final_df)
print(final_df.tail(10))
# save the final dataframe to csv-results folder
csv_results_folder = "csv-results"
if not os.path.isdir(csv_results_folder):
    os.mkdir(csv_results_folder)
csv_filename = os.path.join(csv_results_folder, model_name + ".csv")
final_df.to_csv(csv_filename)
