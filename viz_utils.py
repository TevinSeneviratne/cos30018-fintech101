# viz_utils.py
# Task C.3 visualizations (candlestick + boxplot)
# Uses mplfinance for candlesticks and matplotlib for boxplots.

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import mplcursors  # hover tooltips on MA line(s)

# ---------- helpers ----------

def _to_standard_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to ['Open','High','Low','Close','Volume'].
    Prefer 'adjclose' for 'Close' if present; otherwise use 'close'.
    """
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)

    if "open" in cols:   out["Open"]   = pd.to_numeric(df[cols["open"]], errors="coerce")
    if "high" in cols:   out["High"]   = pd.to_numeric(df[cols["high"]], errors="coerce")
    if "low"  in cols:   out["Low"]    = pd.to_numeric(df[cols["low"]], errors="coerce")
    if "adjclose" in cols:
        out["Close"] = pd.to_numeric(df[cols["adjclose"]], errors="coerce")
    elif "close" in cols:
        out["Close"] = pd.to_numeric(df[cols["close"]], errors="coerce")
    if "volume" in cols:
        out["Volume"] = pd.to_numeric(df[cols["volume"]], errors="coerce")

    required = {"Open", "High", "Low", "Close"}
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Missing columns for OHLC: {missing}. Got: {list(df.columns)}")
    return out


def _aggregate_n_trading_days(ohlc: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Aggregate each block of n *rows* (trading sessions) into one candle.
    Index of each aggregated row = last date in that block.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    g = np.arange(len(ohlc)) // n
    agg = pd.DataFrame({
        "Open":  ohlc["Open"].groupby(g).first(),
        "High":  ohlc["High"].groupby(g).max(),
        "Low":   ohlc["Low"].groupby(g).min(),
        "Close": ohlc["Close"].groupby(g).last(),
    })
    if "Volume" in ohlc.columns:
        agg["Volume"] = ohlc["Volume"].groupby(g).sum()

    last_dates = ohlc.index.to_series().groupby(g).last()
    agg.index = pd.DatetimeIndex(last_dates.values)
    return agg


# ---------- public API ----------

def plot_candles(
    df: pd.DataFrame,
    n: int = 1,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 6),
    ma_window: int = 20,      # moving-average window for blue line
    style: str = "yahoo",
    volume: bool = True,
    hover_ma: bool = True,
    crosshair: bool = False,
):
    """
    Candlestick plot with a thick blue SMA line + legend + hover tooltip.
    IMPORTANT: we do NOT override the x-axis formatter; mplfinance will
    render dates correctly based on the DataFrame's DateTime index.
    """
    ohlc = _to_standard_ohlc(df)
    if n > 1:
        ohlc = _aggregate_n_trading_days(ohlc, n)

    if title is None:
        title = f"Candlesticks ({n}-day candles)"

    # Precompute an SMA series and draw it via addplot so it appears as a Line2D.
    sma = ohlc["Close"].rolling(ma_window).mean()
    ap  = mpf.make_addplot(sma, color="#1f77b4", width=2.5, alpha=1.0,
                           panel=0, label=f"SMA{ma_window}")

    fig, axes = mpf.plot(
        ohlc,
        type="candle",
        addplot=ap,
        volume=volume,
        style=style,
        figsize=figsize,
        title=title,
        tight_layout=True,
        returnfig=True,
    )

    # Price axis (axes can be a list when volume is shown)
    ax_price = axes[0] if isinstance(axes, (list, tuple, np.ndarray)) else axes

    # Optional crosshair
    if crosshair:
        from matplotlib.widgets import Cursor
        Cursor(ax_price, useblit=True, color="gray", linewidth=1)

    # Legend for the SMA line
    lines = [ln for ln in ax_price.lines if (ln.get_label() or "").startswith("SMA")]
    if lines:
        ax_price.legend(handles=lines, loc="upper left")

        # Hover tooltip using *row index* → true date from ohlc.index
        if hover_ma:
            line  = lines[0]
            xdata = np.asarray(line.get_xdata())  # usually 0..N-1 floats
            ydata = np.asarray(line.get_ydata())
            dates = pd.DatetimeIndex(ohlc.index)

            cur = mplcursors.cursor(line, hover=True)

            @cur.connect("add")
            def _on_add(sel, _x=xdata, _y=ydata, _dates=dates, _label=line.get_label()):
                xi = int(np.rint(sel.target[0]))                  # nearest plotted row
                xi = max(0, min(xi, len(_dates) - 1))             # clamp
                val = _y[xi]
                if np.isfinite(val):
                    dt = _dates[xi]
                    sel.annotation.set_text(f"{dt:%Y-%m-%d}\n{_label}: {val:.2f}")
                    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
                else:
                    sel.annotation.set_text("")

    plt.show()


def plot_moving_window_boxplots(
    df: pd.DataFrame,
    column: str = "adjclose",
    window: int = 20,
    stride: int = 5,
    show_outliers: bool = False,
    figsize: tuple[int, int] = (13, 6),
    title: str | None = None,
):
    """
    Boxplots over a sliding window. Each box summarizes `column` across
    `window` consecutive trading days; new boxes every `stride` days.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    cols = {c.lower(): c for c in df.columns}
    col = cols.get(column.lower())
    if col is None:
        raise KeyError(f"Column '{column}' not found in DataFrame: {list(df.columns)}")

    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(series) < window:
        raise ValueError(f"Not enough rows ({len(series)}) for window={window}")

    windows, labels = [], []
    idx = series.index

    for start in range(0, len(series) - window + 1, stride):
        win = series.iloc[start:start + window].to_numpy()
        windows.append(win)
        labels.append(idx[start + window - 1].strftime("%Y-%m-%d"))

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(windows, showfliers=show_outliers)

    n_boxes = len(labels)
    ax.set_xlim(0.5, n_boxes + 0.5)
    if n_boxes <= 15:
        positions, lbls = np.arange(1, n_boxes + 1), labels
    else:
        positions = np.linspace(1, n_boxes, 15, dtype=int)
        lbls = [labels[p - 1] for p in positions]

    ax.set_xticks(positions)
    ax.set_xticklabels(lbls, rotation=45, ha="right")

    if title is None:
        title = f"Rolling {window}-day Boxplots of {column.capitalize()} (stride={stride})"
    ax.set_title(title)
    ax.set_ylabel(column.capitalize())
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
