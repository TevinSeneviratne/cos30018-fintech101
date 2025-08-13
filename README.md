# COS30018 – FinTech101 (Option C) – Task C.1

This repo contains:
- `stock_prediction.py` (v0.1 baseline)
- `p1-baseline/machine-learning/stock-prediction` (P1 baseline)

## How to run

### Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Run v0.1
python stock_prediction.py

### Run P1 baseline
cd p1-baseline/machine-learning/stock-prediction
python train.py     # trains and saves results/*.weights.h5
python test.py      # evaluates, saves plot & CSV

## Notes / fixes
- Using yfinance instead of yahoo_fin.
- MinMaxScaler input shape fix (2-D).
- Keras 3: explicit input shape; checkpoints end with .weights.h5.
- Test: avoid double inverse-scaling; handle suffixed columns (e.g., adjclose_cba.ax).
