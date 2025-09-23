# experiments_task5.py
from __future__ import annotations
import argparse, csv, os

from data_utils import load_and_process_data
from model_utils import build_sequence_model, train_and_evaluate

TICKER      = "CBA.AX"
START, END  = "2015-01-01", "2023-08-01"
ALL_FEATS   = ["adjclose", "volume", "open", "high", "low"]
N_STEPS     = 60
LOOKUP      = 1

def make_model(name: str, input_shape, output_steps: int):
    """
    Keep architectures quick and consistent across scenarios.
    """
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
        # default: compact
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


def run_scenario(scenario: str, k: int, epochs: int, batch: int):
    """
    scenario in {"multistep_univariate", "multivariate_singlestep", "multivariate_multistep"}
    """
    if scenario == "multistep_univariate":
        feats = ["adjclose"]
        target_mode   = "multistep"
        output_steps  = k
    elif scenario == "multivariate_singlestep":
        feats = ALL_FEATS
        target_mode   = "single"
        output_steps  = 1
    elif scenario == "multivariate_multistep":
        feats = ALL_FEATS
        target_mode   = "multistep"
        output_steps  = k
    else:
        raise ValueError("Unknown scenario")

    bundle = load_and_process_data(
        ticker=TICKER, start_date=START, end_date=END,
        feature_columns=feats,
        n_steps=N_STEPS, lookup_step=LOOKUP,
        scale=True, split_by_date=True, test_size=0.2, shuffle=True,
        nan_mode="ffill_bfill",
        cache_dir="cache", force_refresh=False, auto_adjust=True,
        target_col="adjclose", target_mode=target_mode, output_steps=output_steps,
    )

    model = make_model("lstm_2x64", (bundle.X_train.shape[1], bundle.X_train.shape[2]), output_steps=output_steps)

    res = train_and_evaluate(
        model, bundle,
        epochs=epochs, batch_size=batch, patience=6,
        results_dir="results_task5", run_name=f"{scenario}_k{output_steps}",
        plot_pred=False,
    )
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="multivariate_multistep",
                        choices=["multistep_univariate","multivariate_singlestep","multivariate_multistep"])
    parser.add_argument("--k", type=int, default=5, help="horizon for multistep")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    os.makedirs("results_task5", exist_ok=True)
    out_csv = "results_task5/summary_task5.csv"
    header = ["scenario","k","mae","rmse","mape","epochs","batch_size","best_ckpt","final_loss","val_loss"]
    if not os.path.isfile(out_csv):
        with open(out_csv, "w", newline="") as f:
            csv.writer(f).writerow(header)

    res = run_scenario(args.scenario, args.k, args.epochs, args.batch)
    with open(out_csv, "a", newline="") as f:
        row = [
            args.scenario, args.k,
            res["mae"], res["rmse"], res["mape"],
            res["epochs"], res["batch_size"],
            res["best_ckpt"], res["final_loss"], res["val_loss"],
        ]
        csv.writer(f).writerow(row)

    print("\nDone. Wrote:", out_csv)

if __name__ == "__main__":
    main()
