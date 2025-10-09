# experiments_task6.py
from __future__ import annotations
import argparse, os, csv
import numpy as np

from data_utils import load_and_process_data
from model_utils import build_sequence_model, train_and_evaluate
from arima_utils import (
    ArimaConfig, make_adjclose_series_inverse, split_series_for_test,
    forecast_walk_forward_single_step, forecast_walk_forward_multistep
)
from ensemble_utils import best_weight_grid, blend_weighted, overall_metrics, per_horizon_metrics

# ----------- Defaults (same dataset as C.5) -----------
TICKER      = "CBA.AX"
START, END  = "2015-01-01", "2023-08-01"
ALL_FEATS   = ["adjclose", "volume", "open", "high", "low"]
N_STEPS     = 60

# A tiny helper to standardize our DL model choices here
def make_model(name: str, input_shape, output_steps: int):
    if name == "lstm_2x64":
        layers = [
            {"type":"LSTM","units":64,"return_sequences":True,"dropout":0.2},
            {"type":"LSTM","units":64,"dropout":0.2},
        ]
    elif name == "gru_2x64":
        layers = [
            {"type":"GRU","units":64,"return_sequences":True,"dropout":0.2},
            {"type":"GRU","units":64,"dropout":0.2},
        ]
    else:
        layers = [
            {"type":"LSTM","units":64,"return_sequences":True,"dropout":0.2},
            {"type":"LSTM","units":32,"dropout":0.2},
        ]
    return build_sequence_model(
        input_shape=input_shape,
        layers=layers,
        dense_units=1,
        optimizer="adam",
        loss="mean_squared_error",
        output_steps=output_steps,
    )

def run_single_step_ensemble(k_model: str, arima_cfg: ArimaConfig, epochs: int, batch: int):
    """
    DL: multivariate single-step (OHLCV -> next adjclose)
    ARIMA: one-step walk-forward on inverse-scaled adjclose
    Blend: grid search for best weight on test as a simple baseline (you can split test into halves if you want)
    """
    bundle = load_and_process_data(
        ticker=TICKER, start_date=START, end_date=END,
        feature_columns=ALL_FEATS,
        n_steps=N_STEPS, lookup_step=1,
        scale=True, split_by_date=True, test_size=0.2, shuffle=True,
        nan_mode="ffill_bfill",
        cache_dir="cache", force_refresh=False, auto_adjust=True,
        target_col="adjclose", target_mode="single", output_steps=1,
    )
    model = make_model(k_model, (bundle.X_train.shape[1], bundle.X_train.shape[2]), output_steps=1)
    dl_res = train_and_evaluate(
        model, bundle,
        epochs=epochs, batch_size=batch, patience=6,
        results_dir="results_task6", run_name=f"dl_multivariate_singlestep",
        plot_pred=False,
    )

    # ----- ARIMA forecast aligned to test dates (inverse scale) -----
    series_inv = make_adjclose_series_inverse(bundle)
    y_train_series, y_test_series = split_series_for_test(bundle, series_inv)
    ar_preds = forecast_walk_forward_single_step(y_train_series, y_test_series, arima_cfg)

    # ----- Ground truth & DL predictions on inverse scale -----
    # We already computed DL metrics on inverse scale inside train_and_evaluate,
    # but we need the actual arrays to blend. Re-predict using the best weights model (already in memory).
    y_true = model.predict(bundle.X_test, verbose=0).ravel()  # scaled -> must inverse
    # Inverse-transform to real price units
    from model_utils import _inverse_target  # safe import; used only here
    scaler = bundle.column_scaler.get(bundle.target_col)
    y_true = _inverse_target(scaler, bundle.y_test)       # (n,)
    y_dl   = _inverse_target(scaler, model.predict(bundle.X_test, verbose=0).ravel())

    # ----- Blend -----
    w, m = best_weight_grid(y_true, y_dl, ar_preds, weights=(0.0,0.25,0.5,0.75,1.0), metric="rmse")
    y_blend = blend_weighted(y_dl, ar_preds, w)
    m_blend = overall_metrics(y_true, y_blend)

    return {
        "scenario": "ensemble_singlestep",
        "k": 1,  
        "dl_model": k_model,
        "arima_order": arima_cfg.order,
        "seasonal_order": arima_cfg.seasonal_order if arima_cfg.seasonal_order else (),
        "epochs": epochs, "batch_size": batch,
        "w_best": w,
        "dl_mae": float(m["mae"]), "dl_rmse": float(m["rmse"]), "dl_mape": float(m["mape"]),
        "ens_mae": float(m_blend["mae"]), "ens_rmse": float(m_blend["rmse"]), "ens_mape": float(m_blend["mape"]),
    }

def run_multistep_ensemble(k: int, k_model: str, arima_cfg: ArimaConfig, epochs: int, batch: int):
    """
    DL: univariate multistep (adjclose -> vector t+1..t+k)  (you can swap to multivariate_multistep if preferred)
    ARIMA: recursive k-step walk-forward forecasts aligned to DL test dates
    Blend: weight grid search
    """
    bundle = load_and_process_data(
        ticker=TICKER, start_date=START, end_date=END,
        feature_columns=["adjclose"],
        n_steps=N_STEPS, lookup_step=1,
        scale=True, split_by_date=True, test_size=0.2, shuffle=True,
        nan_mode="ffill_bfill",
        cache_dir="cache", force_refresh=False, auto_adjust=True,
        target_col="adjclose", target_mode="multistep", output_steps=k,
    )
    model = make_model(k_model, (bundle.X_train.shape[1], bundle.X_train.shape[2]), output_steps=k)
    dl_res = train_and_evaluate(
        model, bundle,
        epochs=epochs, batch_size=batch, patience=6,
        results_dir="results_task6", run_name=f"dl_univariate_multistep_k{k}",
        plot_pred=False,
    )

    # Ground truth & DL preds (inverse scale)
    from model_utils import _inverse_target
    scaler = bundle.column_scaler.get(bundle.target_col)
    y_true = _inverse_target(scaler, bundle.y_test)                    # (n, k)
    y_dl   = _inverse_target(scaler, model.predict(bundle.X_test, verbose=0))  # (n, k)

    # ARIMA forecasts for the same dates
    series_inv = make_adjclose_series_inverse(bundle)
    y_train_series, y_test_series = split_series_for_test(bundle, series_inv)
    ar_preds = forecast_walk_forward_multistep(y_train_series, y_test_series, k=k, cfg=arima_cfg)  # (n,k)

    # Blend
    w, _ = best_weight_grid(y_true, y_dl, ar_preds, weights=(0.0,0.25,0.5,0.75,1.0), metric="rmse")
    y_blend = blend_weighted(y_dl, ar_preds, w)
    m_dl    = overall_metrics(y_true, y_dl)
    m_blend = overall_metrics(y_true, y_blend)
    per_h   = per_horizon_metrics(y_true, y_blend)

    return {
        "scenario": "ensemble_multistep",
        "k": k, "dl_model": k_model,
        "arima_order": arima_cfg.order,
        "seasonal_order": arima_cfg.seasonal_order if arima_cfg.seasonal_order else (),
        "epochs": epochs, "batch_size": batch,
        "w_best": w,
        "dl_mae": float(m_dl["mae"]), "dl_rmse": float(m_dl["rmse"]), "dl_mape": float(m_dl["mape"]),
        "ens_mae": float(m_blend["mae"]), "ens_rmse": float(m_blend["rmse"]), "ens_mape": float(m_blend["mape"]),
        **{f"metric:{k}": float(v) for k, v in per_h.items()}
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both",
                        choices=["singlestep","multistep","both"])
    parser.add_argument("--k", type=int, default=5, help="horizon for multistep")
    parser.add_argument("--dl", type=str, default="lstm_2x64", choices=["lstm_2x64","gru_2x64","default"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    # ARIMA/SARIMA
    parser.add_argument("--order", type=int, nargs=3, default=[5,1,0], help="ARIMA(p,d,q)")
    parser.add_argument("--seasonal", type=int, nargs=4, default=None, help="SARIMA(P,D,Q,s)")
    args = parser.parse_args()

    os.makedirs("results_task6", exist_ok=True)
    out_csv = "results_task6/summary_task6.csv"
    header = [
        "scenario","k","dl_model","arima_order","seasonal_order","epochs","batch_size",
        "w_best","dl_mae","dl_rmse","dl_mape","ens_mae","ens_rmse","ens_mape"
    ]
    if not os.path.isfile(out_csv):
        with open(out_csv, "w", newline="") as f:
            csv.writer(f).writerow(header)

    cfg = ArimaConfig(order=tuple(args.order),
                      seasonal_order=tuple(args.seasonal) if args.seasonal else None)

    rows = []
    if args.mode in ("singlestep","both"):
        r1 = run_single_step_ensemble(args.dl, cfg, args.epochs, args.batch)
        rows.append(r1)
    if args.mode in ("multistep","both"):
        r2 = run_multistep_ensemble(args.k, args.dl, cfg, args.epochs, args.batch)
        rows.append(r2)

    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([
                r.get("scenario",""),
                r.get("k",""),
                r.get("dl_model",""),
                r.get("arima_order",""),
                r.get("seasonal_order",""),
                r.get("epochs",""),
                r.get("batch_size",""),
                r.get("w_best",""),
                r.get("dl_mae",""), r.get("dl_rmse",""), r.get("dl_mape",""),
                r.get("ens_mae",""), r.get("ens_rmse",""), r.get("ens_mape",""),
            ])

    print("\nDone. Wrote:", out_csv)

if __name__ == "__main__":
    main()
