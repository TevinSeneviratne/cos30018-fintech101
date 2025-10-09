# ensemble_utils.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Iterable, Dict

def blend_weighted(pred_dl: np.ndarray, pred_arima: np.ndarray, w: float) -> np.ndarray:
    """
    Weighted blend of DL and ARIMA predictions.
    Supports shapes: (n,) or (n,k). Returns same shape.
    """
    pred_dl = np.asarray(pred_dl, dtype=float)
    pred_ar = np.asarray(pred_arima, dtype=float)
    return w * pred_dl + (1.0 - w) * pred_ar

def mae(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(np.abs(a - b)))

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mape(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(np.abs((a - b) / np.clip(a, 1e-8, None))) * 100.0)

def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.ndim == 2:
        yt = y_true.reshape(-1,)
        yp = y_pred.reshape(-1,)
    else:
        yt, yp = y_true, y_pred
    return {"mae": mae(yt, yp), "rmse": rmse(yt, yp), "mape": mape(yt, yp)}

def per_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if y_true.ndim != 2 or y_pred.ndim != 2:
        return out
    assert y_true.shape[1] == y_pred.shape[1]
    k = y_true.shape[1]
    for i in range(k):
        out[f"mae@t+{i+1}"]  = mae(y_true[:, i], y_pred[:, i])
        out[f"rmse@t+{i+1}"] = rmse(y_true[:, i], y_pred[:, i])
        out[f"mape@t+{i+1}"] = mape(y_true[:, i], y_pred[:, i])
    return out

def best_weight_grid(y_true: np.ndarray, pred_dl: np.ndarray, pred_arima: np.ndarray,
                     weights: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
                     metric: str = "rmse") -> Tuple[float, Dict[str, float]]:
    """
    Simple grid search over blend weight. Returns (best_w, metrics_for_best).
    """
    best_w, best_val = None, float("inf")
    best_metrics = {}
    for w in weights:
        blended = blend_weighted(pred_dl, pred_arima, w)
        m = overall_metrics(y_true, blended)
        score = m[metric]
        if score < best_val:
            best_val, best_w, best_metrics = score, w, m
    return float(best_w), best_metrics
