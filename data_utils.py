# data_utils.py
# Helper utilities to load, cache, clean, scale and window the stock dataset.
# Designed to be dropped into v0.1 without changing the overall structure too much.

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf


@dataclass
class DataBundle:
    """What we return to the caller (v0.1)."""
    # Arrays (ready to feed into Keras)
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    # Original & test frames (useful for plotting / inspection)
    df: pd.DataFrame
    test_df: pd.DataFrame

    # For inverse scaling later
    column_scaler: Dict[str, MinMaxScaler]

    # For predicting “next” price beyond the dataset
    last_sequence: np.ndarray

    # Housekeeping
    feature_columns: List[str]
    n_steps: int
    lookup_step: int


# ---------- small utilities ----------

def _ensure_cache_dir(cache_dir: Optional[str]) -> Optional[str]:
    if cache_dir is None:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _safe_cache_name(ticker: str, start: str, end: str, auto_adjust: bool) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{ticker}_{start}_{end}_adj{int(auto_adjust)}.csv")
    return safe


def _read_cache(cache_path: str) -> pd.DataFrame:
    """
    Be tolerant reading old/new cache formats:
    - Preferred: a 'Date' column (used as index).
    - Fallback: first column is the date index (no header).
    """
    try:
        df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
    except Exception:
        df = pd.read_csv(cache_path, parse_dates=True, index_col=0)
        if df.index.name is None:
            df.index.name = "Date"
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def _download_yf(ticker: str, start: Optional[str], end: Optional[str], auto_adjust: bool) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    # Make sure we always have a named datetime index
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to lowercase with underscores and ensure we have plain, unsuffixed
    columns for: open, high, low, close, adjclose, volume.

    yfinance can return MultiIndex columns like ('Close','CBA.AX'), which we flatten
    to 'close_cba.ax'. Here we "lift" those back to 'close' (and similarly for the
    other bases). If 'Adj Close' is missing (common with auto_adjust=True), we treat
    'close' as 'adjclose'.
    """
    out = df.copy()

    # Flatten MultiIndex columns if any, then normalize case/spacing
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join([str(x).strip().lower().replace(" ", "_") for x in col if x is not None])
            for col in out.columns
        ]
    else:
        out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    # Unify common variants (handle tokens appearing anywhere in the name)
    def _normalize_name(c: str) -> str:
        c = c.replace("adj_close", "adjclose")
        c = c.replace("close_price", "close")
        return c
    out.columns = [_normalize_name(c) for c in out.columns]

    # Lift suffixed per-ticker columns back to base names
    bases = ["open", "high", "low", "close", "adjclose", "volume"]
    for base in bases:
        if base not in out.columns:
            # Look for 'base_*' (e.g., 'close_cba.ax')
            candidates = [c for c in out.columns if c.startswith(base + "_")]
            if not candidates:
                # very defensive: also allow '*_base' (rare)
                candidates = [c for c in out.columns if c.endswith("_" + base)]
            if candidates:
                out[base] = out[candidates[0]]

    # If still no adjclose but we do have close, treat close as adjclose
    if "adjclose" not in out.columns and "close" in out.columns:
        out["adjclose"] = out["close"]

    # Final sanity
    if "adjclose" not in out.columns and "close" not in out.columns:
        raise KeyError(
            f"Neither 'adjclose' nor 'close' present in downloaded data. Columns: {list(out.columns)}"
        )

    return out



def _handle_nans(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    mode = (mode or "").lower()
    if mode in ("ffill_bfill", "ffill+bbill", "ffill+bfill"):
        df = df.ffill().bfill()
    elif mode in ("drop", "dropna"):
        df = df.dropna()
    elif mode in ("zero", "zeros", "fill0"):
        df = df.fillna(0)
    else:
        # default: forward then backward fill
        df = df.ffill().bfill()
    return df


# ---------- main loader ----------

def load_and_process_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    n_steps: int = 50,
    lookup_step: int = 1,
    scale: bool = True,
    split_by_date: bool = True,
    test_size: float = 0.2,
    shuffle: bool = True,
    nan_mode: str = "ffill_bfill",
    cache_dir: Optional[str] = "cache",
    force_refresh: bool = False,
    auto_adjust: bool = True,
) -> DataBundle:
    """
    Downloads (or loads from cache) a DataFrame, standardizes column names,
    handles NaNs, scales feature columns, and creates (X, y) windows.

    This mirrors the flexibility of the (P1) loader, but returns a simple bundle
    that v0.1 can consume with minimal changes.
    """
    cache_dir = _ensure_cache_dir(cache_dir)
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, _safe_cache_name(ticker, start_date or "None", end_date or "None", auto_adjust))

    # 1) Load
    if cache_path and os.path.isfile(cache_path) and not force_refresh:
        df = _read_cache(cache_path)
    else:
        df = _download_yf(ticker, start_date, end_date, auto_adjust=auto_adjust)
        if cache_path:
            df.to_csv(cache_path)

    # 2) Standardize column names
    df = _standardize_columns(df)

    # 3) Handle NaNs
    df = _handle_nans(df, nan_mode)

    # 4) Decide features
    if not feature_columns:
        feature_columns = ["adjclose", "volume", "open", "high", "low"]

    # normalize user-provided names
    feature_columns = [c.strip().lower() for c in feature_columns]
    cols_present = set(df.columns)
    # If adjclose requested but only close exists
    if "adjclose" in feature_columns and "adjclose" not in cols_present and "close" in cols_present:
        df["adjclose"] = df["close"]

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Requested feature(s) missing from data: {missing}. Available: {list(df.columns)}")

    # 5) Optional scaling
    column_scaler: Dict[str, MinMaxScaler] = {}
    if scale:
        for col in feature_columns:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]].values)
            column_scaler[col] = scaler

    # 6) Build sequences (windows) and labels (future adjclose)
    df["future"] = df["adjclose"].shift(-lookup_step)

    # Keep last lookup window to extrapolate “next” prediction later
    last_sequence_block = df[feature_columns].tail(lookup_step).to_numpy(dtype=np.float32)

    # Drop rows with NaN labels
    df = df.dropna().copy()

    sequences = deque(maxlen=n_steps)
    X, y, dates = [], [], []
    for row_date, row in df.iterrows():
        seq_values = row[feature_columns].values.astype(np.float32)
        sequences.append(seq_values)
        if len(sequences) == n_steps:
            X.append(np.array(sequences, dtype=np.float32))
            y.append(np.float32(row["future"]))
            dates.append(row_date)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # 7) Train/test split
    if split_by_date:
        n_train = int((1.0 - test_size) * len(X))
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]
        if shuffle:
            # shuffle within train/test partitions only
            idx_tr = np.random.permutation(len(X_train))
            idx_te = np.random.permutation(len(X_test))
            X_train, y_train = X_train[idx_tr], y_train[idx_tr]
            X_test, y_test = X_test[idx_te], y_test[idx_te]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    # 8) Build a test_df aligned to X_test (use dates captured above)
    # dates corresponds to the X (after the first n_steps-1 rows). Align to X_test.
    full_test_df = pd.DataFrame(index=pd.DatetimeIndex(dates, name="Date"))
    full_test_df["adjclose"] = df.loc[dates, "adjclose"].values

    # last sequence for “next” prediction
    # take the last n_steps rows from the full df (feature space), then append lookup_step tail
    recent_block = df[feature_columns].tail(n_steps).to_numpy(dtype=np.float32)
    last_sequence = np.vstack([recent_block, last_sequence_block]).astype(np.float32)

    return DataBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        df=df.copy(),
        test_df=full_test_df.copy(),
        column_scaler=column_scaler,
        last_sequence=last_sequence,
        feature_columns=feature_columns,
        n_steps=n_steps,
        lookup_step=lookup_step,
    )
