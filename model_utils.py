# model_utils.py
from __future__ import annotations
import os
from typing import Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -------------------- Reproducibility (optional) --------------------

def set_global_seed(seed: Optional[int] = None) -> None:
    """Set Python/TF seeds if provided."""
    if seed is None:
        return
    try:
        import random, os
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass


# -------------------- Model builder --------------------

def build_sequence_model(
    input_shape: Tuple[int, int],
    layers,
    dense_units: int = 1,
    optimizer: str = "adam",
    loss: str = "mean_squared_error",
    output_steps: int = 1,        # k-step outputs
) -> tf.keras.Model:
    """
    Build a Keras sequence model from a layer spec list. Each layer spec is a dict:
      {"type":"LSTM"/"GRU"/"RNN", "units":64, "return_sequences":True/False,
       "dropout":0.2, "bidirectional":(True/False optional)}
    """
    model = Sequential()
    first = True
    for spec in layers:
        typ = str(spec.get("type", "LSTM")).upper()
        units = int(spec.get("units", 64))
        rs   = bool(spec.get("return_sequences", False))
        do   = float(spec.get("dropout", 0.0))
        bi   = bool(spec.get("bidirectional", False))

        def _mk(layer_cls):
            if first:
                lyr = layer_cls(units, return_sequences=rs, input_shape=input_shape)
            else:
                lyr = layer_cls(units, return_sequences=rs)
            return lyr

        base = LSTM
        if typ == "GRU":
            base = GRU
        elif typ in ("RNN", "SIMPLERNN"):
            base = SimpleRNN

        lyr = _mk(base)
        if bi:
            lyr = Bidirectional(lyr)
        model.add(lyr)
        if do > 0:
            model.add(Dropout(do))
        first = False

    # Output dense: for multistep k>1, output a vector; else 1 (dense_units kept for compatibility)
    model.add(Dense(output_steps if output_steps > 1 else dense_units))
    model.compile(optimizer=optimizer, loss=loss)
    return model


# -------------------- Helpers --------------------

def _inverse_target(scaler, arr: np.ndarray) -> np.ndarray:
    """
    Inverse transform predictions/targets that may be:
      - shape (n,) single step
      - shape (n, 1) single step (model output)
      - shape (n, k) multistep
    """
    if scaler is None:
        return np.array(arr, dtype=float)

    arr = np.asarray(arr)
    # Single-step, already 1D
    if arr.ndim == 1:
        return scaler.inverse_transform(arr.reshape(-1, 1)).ravel()

    # 2D cases
    n, m = arr.shape
    if m == 1:
        # Keep single-step as 1D vector
        return scaler.inverse_transform(arr.reshape(-1, 1)).ravel()

    # Multistep: inverse each horizon separately and stack as (n, k)
    cols = []
    for i in range(m):
        col = scaler.inverse_transform(arr[:, i].reshape(-1, 1)).ravel()
        cols.append(col)
    return np.stack(cols, axis=1)


def _overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """If multistep (n, k), flatten to compute global MAE/RMSE/MAPE across all horizons."""
    if y_pred.ndim == 2:
        actual = y_true.reshape(-1,)
        pred   = y_pred.reshape(-1,)
    else:
        actual = y_true.astype(float)
        pred   = y_pred.astype(float)

    mae  = float(mean_absolute_error(actual, pred))
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mape = float(np.mean(np.abs((actual - pred) / np.clip(actual, 1e-8, None))) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _per_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Returns horizon-wise MAE/RMSE/MAPE for multistep only.
    Empty dict for single-step or if shapes aren't 2D.
    """
    out: Dict[str, float] = {}
    if y_true.ndim != 2 or y_pred.ndim != 2:
        return out
    if y_true.shape[1] != y_pred.shape[1]:
        return out
    k = y_pred.shape[1]
    for i in range(k):
        a = y_true[:, i].ravel()
        p = y_pred[:, i].ravel()
        out[f"mae@t+{i+1}"]  = float(mean_absolute_error(a, p))
        out[f"rmse@t+{i+1}"] = float(np.sqrt(mean_squared_error(a, p)))
        out[f"mape@t+{i+1}"] = float(np.mean(np.abs((a - p) / np.clip(a, 1e-8, None))) * 100.0)
    return out


# -------------------- Train & Evaluate --------------------

def train_and_evaluate(
    model: tf.keras.Model,
    bundle,
    *,
    epochs: int = 25,
    batch_size: int = 32,
    patience: int = 6,
    results_dir: str = "results",
    run_name: str = "run",
    plot_pred: bool = False,   # kept for compatibility; if True, writes a PNG
) -> Dict[str, float]:
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, f"{run_name}.weights.h5")

    cbs = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, save_weights_only=True),
    ]

    hist = model.fit(
        bundle.X_train, bundle.y_train,
        validation_data=(bundle.X_test, bundle.y_test),
        epochs=epochs, batch_size=batch_size, callbacks=cbs, verbose=1
    )

    # Predict on test
    y_pred = model.predict(bundle.X_test, verbose=0)

    # Inverse-scale using target scaler if available
    scaler = bundle.column_scaler.get(bundle.target_col, None)
    y_true_inv = _inverse_target(scaler, bundle.y_test)
    y_pred_inv = _inverse_target(scaler, y_pred)

    # Metrics
    overall = _overall_metrics(y_true_inv, y_pred_inv)
    per_h = _per_horizon_metrics(y_true_inv, y_pred_inv)

    # Optional plot (single sample) for quick visual check (multistep only)
    pred_plot_path = ""
    if plot_pred and y_pred_inv.ndim == 2 and y_pred_inv.shape[1] >= 1:
        try:
            import matplotlib.pyplot as plt
            idx = -1  # last test sample
            k = y_pred_inv.shape[1]
            plt.figure()
            plt.plot(range(1, k+1), y_true_inv[idx], label="true")
            plt.plot(range(1, k+1), y_pred_inv[idx], label="pred")
            plt.xlabel("Horizon (days)")
            plt.ylabel("Price")
            plt.title(f"Forecast (sample {idx}) — {run_name}")
            plt.legend()
            pred_plot_path = os.path.join(results_dir, f"{run_name}_forecast.png")
            plt.savefig(pred_plot_path, bbox_inches="tight")
            plt.close()
        except Exception:
            pred_plot_path = ""

    res = {
        "run_name":   run_name,
        "mae":        overall["mae"],
        "rmse":       overall["rmse"],
        "mape":       overall["mape"],
        "epochs":     int(len(hist.history["loss"])),
        "batch_size": int(batch_size),
        "best_ckpt":  ckpt_path,
        "final_loss": float(hist.history["loss"][-1]),
        "val_loss":   float(min(hist.history["val_loss"])),
        "pred_plot":  pred_plot_path,
    }
    # Non-breaking extra details (caller can ignore)
    res.update({f"metric:{k}": v for k, v in per_h.items()})
    return res
