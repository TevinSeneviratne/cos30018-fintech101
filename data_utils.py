# data_utils.py
# Helper utilities to load, cache, clean, scale and window the stock dataset.
# Supports multistep targets (k-step forecasting) and multivariate inputs.
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf


@dataclass
class DataBundle:
    """Return object the training/eval code can use directly."""
    # Arrays (ready for Keras)
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    # Original/test frames for inspection/plot
    df: pd.DataFrame
    test_df: pd.DataFrame

    # For inverse scaling later
    column_scaler: Dict[str, MinMaxScaler]

    # For predicting “next” sequence beyond the dataset
    last_sequence: np.ndarray

    # Housekeeping
    feature_columns: List[str]
    n_steps: int
    lookup_step: int
    output_steps: int           # how many future steps we predict
    target_col: str             # usually 'adjclose'
    target_mode: str            # 'single' or 'multistep'


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
    """Read old/new cache formats (yfinance may write different headers)."""
    try:
        # Try MultiIndex header first
        df = pd.read_csv(cache_path, header=[0, 1], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(
                    str(x).strip().lower().replace(" ", "_")
                    for x in tup if x and str(x).lower() != "nan"
                )
                for tup in df.columns
            ]
    except Exception:
        try:
            df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        except Exception:
            df = pd.read_csv(cache_path, parse_dates=True, index_col=0)

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df.index.name = "Date"
    return df


def _download_yf(ticker: str, start: Optional[str], end: Optional[str], auto_adjust: bool) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize to lower-case OHLCV schema:
    ['open','high','low','close','adjclose','volume']
    """
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception:
            pass

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join([str(x) for x in tup if x is not None]).strip()
                       for tup in out.columns]

    cols_lower = [str(c).strip().lower() for c in out.columns]

    def base_of(name: str) -> str | None:
        name = name.replace("adj close", "adjclose")
        if "adjclose" in name: return "adjclose"
        if "close"    in name: return "close"
        if "open"     in name: return "open"
        if "high"     in name: return "high"
        if "low"      in name: return "low"
        if "volume"   in name: return "volume"
        return None

    remap = {}
    for orig, lc in zip(out.columns, cols_lower):
        base = base_of(lc)
        if base and orig not in remap:
            remap[orig] = base

    out = out.rename(columns=remap)
    wanted = ["open", "high", "low", "close", "adjclose", "volume"]
    present = [c for c in wanted if c in out.columns]
    out = out[present]

    if ("adjclose" not in out.columns) and ("close" not in out.columns):
        title_map = {"Open": "open", "High": "high", "Low": "low",
                     "Close": "close", "Adj Close": "adjclose", "Volume": "volume"}
        found = [c for c in df.columns if c in title_map]
        if found:
            tmp = df[found].rename(columns=title_map)
            out = tmp[[c for c in wanted if c in tmp.columns]]

        if out.shape[1] == 0:
            raise KeyError(
                "Neither 'adjclose' nor 'close' present after standardization. "
                f"Original columns: {list(df.columns)}"
            )

    if "adjclose" not in out.columns and "close" in out.columns:
        out["adjclose"] = out["close"]

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out.sort_index(inplace=True)
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
        df = df.ffill().bfill()
    return df


def _build_multistep_targets(series: pd.Series, k: int) -> pd.DataFrame:
    """
    Given a 1D series, build a DataFrame with columns:
      t+1, t+2, ..., t+k
    where each col is series shifted by -step.
    """
    if k < 1:
        raise ValueError("output_steps (k) must be >= 1")
    out = {}
    for step in range(1, k + 1):
        out[f"t+{step}"] = series.shift(-step)
    return pd.DataFrame(out)


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
    *,
    target_col: str = "adjclose",
    target_mode: str = "single",       # 'single' or 'multistep'
    output_steps: int = 1,              # k: number of future days (multistep)
) -> DataBundle:
    """
    Downloads (or loads from cache), standardizes columns, handles NaNs,
    fits scalers on TRAIN ONLY, scales features, and creates (X, y) windows.

    - If target_mode == 'single': y is a scalar (series shifted by -lookup_step)
    - If target_mode == 'multistep': y is a vector of length `output_steps`
      (next 1..k days). lookup_step is ignored for multistep.
    """
    cache_dir = _ensure_cache_dir(cache_dir)
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(
            cache_dir, _safe_cache_name(ticker, start_date or "None", end_date or "None", auto_adjust)
        )

    # 1) Load
    if cache_path and os.path.isfile(cache_path) and not force_refresh:
        df = _read_cache(cache_path)
        df = _standardize_columns(df)
    else:
        df = _download_yf(ticker, start_date, end_date, auto_adjust=auto_adjust)
        df = _standardize_columns(df)
        if cache_path:
            df.to_csv(cache_path)

    # 2) NaNs
    df = _handle_nans(df, nan_mode)

    # 3) Decide features
    if not feature_columns:
        feature_columns = ["adjclose", "volume", "open", "high", "low"]
    feature_columns = [c.strip().lower() for c in feature_columns]

    if "adjclose" in feature_columns and "adjclose" not in df.columns and "close" in df.columns:
        df["adjclose"] = df["close"]

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Requested feature(s) missing: {missing}. Available: {list(df.columns)}")

    # ---------- Determine train/test cutoff (using unscaled data) ----------
    target_col = target_col.lower()
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found; available: {list(df.columns)}")

    # Build a provisional df_y (UNSCALED) just to locate the cutoff date.
    if target_mode == "multistep":
        provisional_y = _build_multistep_targets(df[target_col], output_steps)
        df_y_prov = pd.concat([df, provisional_y], axis=1).dropna().copy()
    else:
        df_y_prov = df.copy()
        df_y_prov["future"] = df_y_prov[target_col].shift(-lookup_step)
        df_y_prov = df_y_prov.dropna().copy()

    if split_by_date:
        n_samples_total = len(df_y_prov)            # number of labelable rows (== number of windows)
        n_train_samples = int((1.0 - test_size) * n_samples_total)
        n_train_samples = max(1, min(n_train_samples, n_samples_total - 1))
        cutoff_date = df_y_prov.index[n_train_samples - 1]
    else:
        cutoff_date = None  # not used in random split

    # 4) Scaling (FIT ON TRAIN ONLY)
    column_scaler: Dict[str, MinMaxScaler] = {}
    if scale:
        if split_by_date:
            train_slice = df.loc[:cutoff_date, feature_columns]
        else:
            # If we'll use random split later, approximate by fitting on the first (1-test_size) proportion.
            approx_cut = int((1.0 - test_size) * len(df))
            train_slice = df.iloc[:max(1, approx_cut)][feature_columns]

        for col in feature_columns:
            scaler = MinMaxScaler()
            scaler.fit(train_slice[[col]].values)
            df[col] = scaler.transform(df[[col]].values)
            column_scaler[col] = scaler

    # 5) Build FINAL targets from the (possibly) SCALED df
    if target_mode == "multistep":
        y_frame = _build_multistep_targets(df[target_col], output_steps)
        df_y = pd.concat([df, y_frame], axis=1).dropna().copy()
        target_cols = [f"t+{i}" for i in range(1, output_steps + 1)]
    else:
        df_y = df.copy()
        df_y["future"] = df_y[target_col].shift(-lookup_step)
        df_y = df_y.dropna().copy()
        target_cols = ["future"]

    # 6) Build sequences (windows) and labels
    sequences = deque(maxlen=n_steps)
    X, y, dates = [], [], []
    for row_date, row in df_y.iterrows():
        seq_values = row[feature_columns].values.astype(np.float32)
        sequences.append(seq_values)
        if len(sequences) == n_steps:
            X.append(np.array(sequences, dtype=np.float32))
            y.append(row[target_cols].values.astype(np.float32))
            dates.append(row_date)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    # If single, make y 1D (n,) for convenience
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1,)

    # 7) Train/test split (keep TEST chronological; shuffle TRAIN only)
    if split_by_date:
        n_train = int((1.0 - test_size) * len(X))
        n_train = max(1, min(n_train, len(X) - 1))
        X_train, y_train, dates_train = X[:n_train], y[:n_train], np.array(dates[:n_train])
        X_test,  y_test,  dates_test  = X[n_train:], y[n_train:], np.array(dates[n_train:])

        if shuffle:
            idx_tr = np.random.permutation(len(X_train))
            X_train, y_train, dates_train = X_train[idx_tr], y_train[idx_tr], dates_train[idx_tr]
        # DO NOT shuffle test
    else:
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X, X, y, y, dates, dates, test_size=test_size, shuffle=True
        )

    # 8) Test frame aligned to X_test order only
    test_df = pd.DataFrame(index=pd.DatetimeIndex(dates_test, name="Date"))
    test_df[target_col] = df_y.loc[dates_test, target_col].values

    # 9) last sequence for “next” prediction (use last n_steps feature rows)
    recent_block = df[feature_columns].tail(n_steps).to_numpy(dtype=np.float32)
    last_sequence = recent_block.copy()  # shape (n_steps, n_features)

    return DataBundle(
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        df=df.copy(),    test_df=test_df.copy(),
        column_scaler=column_scaler,
        last_sequence=last_sequence,
        feature_columns=feature_columns,
        n_steps=n_steps,
        lookup_step=lookup_step,
        output_steps=output_steps,
        target_col=target_col,
        target_mode=target_mode,
    )
