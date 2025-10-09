# arima_utils.py
from __future__ import annotations
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dataclasses import dataclass

@dataclass
class ArimaConfig:
    order: Tuple[int, int, int] = (5, 1, 0)      # (p,d,q)
    seasonal_order: Optional[Tuple[int, int, int, int]] = None  # (P,D,Q,s) or None
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True

# ---------------- helpers ----------------

def _inverse_series(series_scaled: pd.Series, scaler) -> pd.Series:
    inv = scaler.inverse_transform(series_scaled.values.reshape(-1, 1)).ravel()
    return pd.Series(inv, index=series_scaled.index, name=series_scaled.name)

def make_adjclose_series_inverse(bundle) -> pd.Series:
    s = bundle.df["adjclose"].copy()
    scaler = bundle.column_scaler.get("adjclose", None)
    if scaler is None:
        return s.astype(float)
    return _inverse_series(s, scaler).astype(float)

def split_series_for_test(bundle, series_inv: pd.Series) -> tuple[pd.Series, pd.Series]:
    first_test_date = bundle.test_df.index[0]
    train_series = series_inv.loc[: first_test_date].iloc[:-1]  # up to day before test start
    test_series  = series_inv.loc[bundle.test_df.index]
    return train_series, test_series

def _coerce_index_for_statsmodels(y: pd.Series) -> pd.Series:
    """
    Statsmodels works best with a DatetimeIndex that has a freq, or a plain RangeIndex.
    Try to set a frequency; if we can't, fall back to RangeIndex (positional).
    """
    y = y.copy()
    if isinstance(y.index, pd.DatetimeIndex):
        freq = pd.infer_freq(y.index)
        if freq:
            y = y.asfreq(freq)
            return y
        # fallback to business day if dense
        try:
            y = y.asfreq("B")
            return y
        except Exception:
            pass
    # Last resort: RangeIndex
    y.index = pd.RangeIndex(start=0, stop=len(y), step=1)
    return y

def _fit_model(y_train: pd.Series, cfg: ArimaConfig):
    y_train = _coerce_index_for_statsmodels(y_train)
    if cfg.seasonal_order:
        model = SARIMAX(
            y_train, order=cfg.order, seasonal_order=cfg.seasonal_order,
            enforce_stationarity=cfg.enforce_stationarity,
            enforce_invertibility=cfg.enforce_invertibility
        )
    else:
        model = ARIMA(y_train, order=cfg.order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit()
    return res

# ---------------- forecasting ----------------

def forecast_walk_forward_single_step(y_train: pd.Series, y_test: pd.Series, cfg: ArimaConfig) -> np.ndarray:
    """
    Walk-forward one-step-ahead forecasting across the test period.
    Re-fits each step for robustness. Returns array shape (len(y_test),).
    """
    history = y_train.copy()
    preds = []
    for t in range(len(y_test)):
        res = _fit_model(history, cfg)
        fc = res.forecast(steps=1)
        # IMPORTANT: use positional access, not label [-1]
        yhat = float(fc.iloc[-1]) if hasattr(fc, "iloc") else float(np.asarray(fc)[-1])
        preds.append(yhat)
        # append the ground-truth observation at this test time
        next_true = y_test.iloc[t]
        # keep history index type consistent (use concat not deprecated append)
        history = pd.concat([history, pd.Series([next_true], index=[y_test.index[t]])])
    return np.asarray(preds, dtype=float)

def forecast_walk_forward_multistep(y_train: pd.Series, y_test: pd.Series, k: int, cfg: ArimaConfig) -> np.ndarray:
    """
    For each test point, produce a k-step recursive forecast using the model
    fitted on history up to that point. Returns shape (n_test, k).
    """
    history = y_train.copy()
    n_test = len(y_test)
    preds = np.zeros((n_test, k), dtype=float)

    for t in range(n_test):
        res = _fit_model(history, cfg)
        fc = res.forecast(steps=k)
        # positional access to ensure robustness
        if hasattr(fc, "iloc"):
            preds[t, :] = np.asarray(fc.iloc[:k], dtype=float)
        else:
            preds[t, :] = np.asarray(fc, dtype=float)[:k]
        # roll history forward with the true observation at t
        true_val = y_test.iloc[t]
        history = pd.concat([history, pd.Series([true_val], index=[y_test.index[t]])])

    return preds
