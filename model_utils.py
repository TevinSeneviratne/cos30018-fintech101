# model_utils.py
# Generic model builder + train/eval utilities for Task C.4 (Keras/TensorFlow)

from __future__ import annotations
import os, time, math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------- model factory ---------

def build_sequence_model(
    input_shape: Tuple[int, int],
    layers: List[Dict[str, Any]],
    dense_units: int = 1,
    dense_activation: Optional[str] = None,
    optimizer: str = "adam",
    loss: str = "mean_squared_error",
):
    """
    Build a Sequential model from a simple layer spec list.

    Example `layers`:
    [
      {"type": "LSTM", "units": 64, "return_sequences": True, "dropout": 0.2},
      {"type": "LSTM", "units": 64, "dropout": 0.2},
      # OR bidirectional:
      {"type": "GRU", "units": 64, "bidirectional": True, "return_sequences": False}
    ]
    """
    model = Sequential()

    for i, spec in enumerate(layers):
        ltype = (spec.get("type") or "LSTM").upper()
        units = int(spec.get("units", 50))
        rseq  = bool(spec.get("return_sequences", i < len(layers)-1))
        dr    = float(spec.get("dropout", 0.0))
        bidi  = bool(spec.get("bidirectional", False))

        # first recurrent layer defines input shape
        kwargs = {}
        if i == 0:
            kwargs["input_shape"] = input_shape

        if ltype == "LSTM":
            base = LSTM(units=units, return_sequences=rseq, **kwargs)
        elif ltype == "GRU":
            base = GRU(units=units, return_sequences=rseq, **kwargs)
        elif ltype in ("RNN", "SIMPLERNN"):
            base = SimpleRNN(units=units, return_sequences=rseq, **kwargs)
        else:
            raise ValueError(f"Unknown layer type: {ltype}")

        layer = Bidirectional(base) if bidi else base
        model.add(layer)

        if dr and dr > 0:
            model.add(Dropout(dr))

    model.add(Dense(dense_units, activation=dense_activation))
    model.compile(optimizer=optimizer, loss=loss)
    return model

# --------- train & evaluate ---------

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _inverse_adjclose(pred_scaled: np.ndarray, y_scaled: np.ndarray, column_scaler: dict):
    """Inverse-scale predictions/targets using 'adjclose' scaler when available."""
    if "adjclose" in column_scaler:
        sc = column_scaler["adjclose"]
        pred = sc.inverse_transform(pred_scaled).ravel()
        true = sc.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    else:
        pred = pred_scaled.ravel()
        true = y_scaled.ravel()
    return true.astype(float), pred.astype(float)

def train_and_evaluate(
    model,
    bundle,
    *,
    epochs: int = 25,
    batch_size: int = 32,
    patience: Optional[int] = 5,
    results_dir: str = "results",
    run_name: Optional[str] = None,
    plot_pred: bool = True,
):
    """
    Train on bundle.(X_train, y_train) and evaluate on bundle.(X_test, y_test).
    Saves best weights and returns a metrics dict.
    """
    _ensure_dir(results_dir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = run_name or f"{ts}"
    ckpt_path = os.path.join(results_dir, f"{run_name}.weights.h5")

    callbacks = []
    if patience and patience > 0:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
    callbacks.append(ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, save_weights_only=True))

    history = model.fit(
        bundle.X_train, bundle.y_train,
        validation_data=(bundle.X_test, bundle.y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Predict on test set
    y_pred_scaled = model.predict(bundle.X_test, verbose=0)
    y_true, y_pred = _inverse_adjclose(y_pred_scaled, bundle.y_test, bundle.column_scaler)

    # Metrics
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    metrics = {
        "run_name": run_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "best_ckpt": ckpt_path,
        "final_loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history.get("val_loss", [math.nan])[-1]),
    }

    # Optional plot
    if plot_pred:
        plt.figure(figsize=(8,4))
        plt.plot(y_true, label="Actual", color="black", linewidth=1)
        plt.plot(y_pred, label="Predicted", linewidth=1.5)
        plt.title(f"Test Predictions — {run_name}\nMAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%")
        plt.xlabel("Time"); plt.ylabel("Price")
        plt.legend(); plt.tight_layout()
        fig_path = os.path.join(results_dir, f"{run_name}_pred.png")
        plt.savefig(fig_path, dpi=140)
        plt.show()
        metrics["pred_plot"] = fig_path

    # Write a tiny metrics file per run
    with open(os.path.join(results_dir, f"{run_name}_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    return metrics
