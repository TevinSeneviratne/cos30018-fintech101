# experiments_task4.py
from __future__ import annotations
import argparse
import csv
import os
import sys
from datetime import datetime

from data_utils import load_and_process_data
from model_utils import build_sequence_model, train_and_evaluate


TICKER      = "CBA.AX"
START, END  = "2015-01-01", "2023-08-01"
FEATURES    = ["adjclose", "volume", "open", "high", "low"]
N_STEPS     = 60
LOOKUP      = 1

# name, layers, epochs, batch
CONFIGS = [
    ("lstm_2x64", [
        {"type": "LSTM", "units": 64, "return_sequences": True,  "dropout": 0.2},
        {"type": "LSTM", "units": 64, "return_sequences": False, "dropout": 0.2},
    ], 25, 32),

    ("gru_2x64", [
        {"type": "GRU",  "units": 64, "return_sequences": True,  "dropout": 0.2},
        {"type": "GRU",  "units": 64, "return_sequences": False, "dropout": 0.2},
    ], 25, 32),

    ("bilstm_1x64", [
        {"type": "LSTM", "units": 64, "bidirectional": True, "return_sequences": False, "dropout": 0.2},
    ], 25, 32),

    ("rnn_2x64", [
        {"type": "RNN",  "units": 64, "return_sequences": True,  "dropout": 0.2},
        {"type": "RNN",  "units": 64, "return_sequences": False, "dropout": 0.2},
    ], 15, 32),

    ("lstm_deeper", [
        {"type": "LSTM", "units": 128, "return_sequences": True,  "dropout": 0.2},
        {"type": "LSTM", "units": 64,  "return_sequences": True,  "dropout": 0.2},
        {"type": "LSTM", "units": 32,  "return_sequences": False, "dropout": 0.2},
    ], 30, 64),
]


def build_arg_parser():
    p = argparse.ArgumentParser(description="Task C.4 model sweeps")
    p.add_argument(
        "--run",
        default="all",
        choices=["all"] + [c[0] for c in CONFIGS],
        help="Which model to run (default: all)"
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Also generate & save a prediction plot for each run"
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Open the saved prediction plot after each run (if --plot is used)"
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Where to store metrics, checkpoints, and plots (default: results)"
    )
    return p


def load_bundle():
    return load_and_process_data(
        ticker=TICKER,
        start_date=START,
        end_date=END,
        feature_columns=FEATURES,
        n_steps=N_STEPS,
        lookup_step=LOOKUP,
        scale=True,
        split_by_date=True,
        test_size=0.2,
        shuffle=True,
        nan_mode="ffill_bfill",
        cache_dir="cache",
        force_refresh=False,
        auto_adjust=True,
    )


def _open_file(path: str):
    """Best-effort open of a file on macOS/Linux/Windows."""
    try:
        if sys.platform == "darwin":
            os.system(f'open "{path}"')
        elif sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            os.system(f'xdg-open "{path}"')
    except Exception:
        pass


def run_single(name: str, layers, epochs: int, batch: int, bundle, results_dir: str, plot_pred: bool, show_plot: bool):
    print("\n==============================")
    print(f"Running: {name}")
    print("==============================")

    model = build_sequence_model(
        input_shape=(bundle.X_train.shape[1], bundle.X_train.shape[2]),
        layers=layers,
        dense_units=1,
        optimizer="adam",
        loss="mean_squared_error",
    )

    # NOTE: train_and_evaluate() in your project does NOT accept show_plot.
    # We only pass plot_pred here.
    res = train_and_evaluate(
        model,
        bundle,
        epochs=epochs,
        batch_size=batch,
        patience=6,
        results_dir=results_dir,
        run_name=name,
        plot_pred=plot_pred,   # save plot if True (your function handles saving)
    )

    # Ensure standard fields for summary
    res.setdefault("run_name", name)
    res.setdefault("epochs", epochs)
    res.setdefault("batch_size", batch)
    res.setdefault("pred_plot", "")  # path to saved png if any

    # If user asked to show, try to open the saved plot
    if show_plot and plot_pred:
        plot_path = res.get("pred_plot")
        if isinstance(plot_path, str) and os.path.isfile(plot_path):
            _open_file(plot_path)

    return res


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # CSV summary file
    summary_path = os.path.join(args.results_dir, "summary.csv")
    header = ["run_name", "mae", "rmse", "mape", "epochs", "batch_size", "best_ckpt", "final_loss", "val_loss", "pred_plot"]
    if not os.path.isfile(summary_path):
        with open(summary_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # load data once
    bundle = load_bundle()

    # choose models to run
    to_run = CONFIGS if args.run == "all" else [next(c for c in CONFIGS if c[0] == args.run)]

    rows = []
    for name, layers, epochs, batch in to_run:
        res = run_single(
            name=name,
            layers=layers,
            epochs=epochs,
            batch=batch,
            bundle=bundle,
            results_dir=args.results_dir,
            plot_pred=args.plot,
            show_plot=args.show,   # handled locally (we don't pass to train_and_evaluate)
        )
        rows.append(res)

        # append to CSV
        with open(summary_path, "a", newline="") as f:
            csv.writer(f).writerow([res.get(k, "") for k in header])

    # pretty print summary table (optional)
    try:
        import pandas as pd
        df = pd.DataFrame([{k: r.get(k, "") for k in header} for r in rows])
        print("\nSummary:")
        print(df.to_string(index=False))
        # also save a timestamped copy for traceability
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(args.results_dir, f"task_c4_summary_{stamp}.csv"), index=False)
    except Exception:
        pass

    print(f"\nAll done. See {summary_path}")


if __name__ == "__main__":
    main()
