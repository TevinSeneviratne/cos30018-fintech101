# experiments_task7.py
from __future__ import annotations
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from data_utils import load_and_process_data
from sentiment_utils import fetch_finnews_yahoo, SentimentScorer, merge_with_prices

# ---------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------
TICKER = "CBA.AX"
START, END = "2018-01-01", "2023-08-01"
CACHE = "cache"

# Results directory (single folder)
RESULTS_DIR = "results_task7"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Data & Sentiment
# ---------------------------------------------------------------------
print("Fetching price data...")
bundle = load_and_process_data(
    ticker=TICKER, start_date=START, end_date=END,
    feature_columns=["adjclose", "open", "high", "low", "volume"],
    n_steps=1, lookup_step=1, scale=False, split_by_date=True,
    test_size=0.2, shuffle=False, nan_mode="ffill_bfill",
    cache_dir=CACHE, force_refresh=False, auto_adjust=True,
)

print("Fetching news headlines...")
df_news = fetch_finnews_yahoo(TICKER, START, END)
scorer = SentimentScorer()
df_sent = scorer.daily_aggregate(df_news)

# Merge sentiment with stock prices
df_full = merge_with_prices(bundle.df, df_sent)
df_full["return"] = df_full["adjclose"].pct_change()
df_full["target"] = (df_full["adjclose"].shift(-1) > df_full["adjclose"]).astype(int)
df_full.dropna(inplace=True)

# ---------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------
# Technical indicators (simple)
df_full["ma5"] = df_full["adjclose"].rolling(5).mean()
df_full["ma10"] = df_full["adjclose"].rolling(10).mean()
df_full["rsi14"] = 100 - (100 / (1 + (df_full["adjclose"].pct_change().clip(lower=-1, upper=1)
                                      .add(1).rolling(14)
                                      .apply(lambda x: (x > 1).sum() / (x <= 1).sum() + 1, raw=False))))
df_full.fillna(method="bfill", inplace=True)

feature_cols_base = ["adjclose", "volume", "ma5", "ma10", "rsi14"]
feature_cols_sent = feature_cols_base + ["sentiment"]

X_base = df_full[feature_cols_base].values
X_sent = df_full[feature_cols_sent].values
y = df_full["target"].values

Xb_train, Xb_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, shuffle=False)
Xs_train, Xs_test, _, _ = train_test_split(X_sent, y, test_size=0.2, shuffle=False)

sc = StandardScaler()
Xb_train = sc.fit_transform(Xb_train); Xb_test = sc.transform(Xb_test)
Xs_train = sc.fit_transform(Xs_train); Xs_test = sc.transform(Xs_test)

# ---------------------------------------------------------------------
# 3. Modelling
# ---------------------------------------------------------------------
def train_and_eval(Xtr, Xte, ytr, yte, tag):
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    metrics = {
        "Accuracy": accuracy_score(yte, ypred),
        "Precision": precision_score(yte, ypred),
        "Recall": recall_score(yte, ypred),
        "F1": f1_score(yte, ypred),
        "ConfMat": confusion_matrix(yte, ypred)
    }
    print(f"\n=== {tag} ===")
    print(classification_report(yte, ypred, digits=3))
    print("Confusion Matrix:\n", metrics["ConfMat"])
    return clf, metrics


clf_base, m_base = train_and_eval(Xb_train, Xb_test, y_train, y_test, "Baseline (No Sentiment)")
clf_sent, m_sent = train_and_eval(Xs_train, Xs_test, y_train, y_test, "With Sentiment")

# ---------------------------------------------------------------------
# 4. Comparison & Saving Results
# ---------------------------------------------------------------------
cmp = pd.DataFrame([m_base, m_sent], index=["Baseline", "WithSentiment"])
cmp_simple = cmp[["Accuracy", "Precision", "Recall", "F1"]]

# Save CSV inside results_task7/
csv_path = os.path.join(RESULTS_DIR, "summary_task7.csv")
cmp_simple.to_csv(csv_path, index=True)
print(f"\nSaved metrics → {csv_path}")

# Plot comparison
plt.figure(figsize=(8, 5))
cmp_simple.plot(kind="bar", rot=0)
plt.title("Performance Comparison: With vs Without Sentiment")
plt.tight_layout()

# Save figure inside results_task7/
plot_path = os.path.join(RESULTS_DIR, "metrics_task7.png")
plt.savefig(plot_path, bbox_inches="tight")
plt.show()

print(f"✅ All results saved to folder: {RESULTS_DIR}")
