# sentiment_utils.py
from __future__ import annotations
import os, pandas as pd, numpy as np
import yfinance as yf
from transformers import pipeline
import datetime as dt
import requests, re

# ------------------ Sentiment Fetchers ------------------

def fetch_finnews_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Pulls recent Yahoo Finance news headlines for a given ticker and date range.
    (Simple free web API via yfinance 'news' attribute fallback.)
    Returns DataFrame: ['date','title','link'].
    """
    data = []
    try:
        yobj = yf.Ticker(ticker)
        for art in yobj.news:
            ts = dt.datetime.fromtimestamp(art["providerPublishTime"])
            if ts.strftime("%Y-%m-%d") >= start and ts.strftime("%Y-%m-%d") <= end:
                data.append({"date": ts.date(), "title": art["title"], "link": art.get("link", "")})
    except Exception:
        pass
    return pd.DataFrame(data)

# ------------------ Sentiment Scoring ------------------

class SentimentScorer:
    """
    Finance-specific sentiment analyzer using FinBERT.
    Aggregates to daily mean compound sentiment ∈ [-1,1].
    """
    def __init__(self, model_name="ProsusAI/finbert"):
        self.pipe = pipeline("sentiment-analysis", model=model_name, framework="pt")

    def score_texts(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        preds = self.pipe(texts, truncation=True, batch_size=8)
        out = []
        for p in preds:
            label = p["label"].lower()
            score = p["score"]
            if label == "positive":
                out.append(score)
            elif label == "negative":
                out.append(-score)
            else:
                out.append(0.0)
        return out

    def daily_aggregate(self, df_news: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with ['date','sentiment'].
        Also saves a copy under results_task7/sentiment_daily.csv for inspection.
        """
        if df_news.empty:
            return pd.DataFrame(columns=["date", "sentiment"])

        df = df_news.copy()
        df["sentiment_score"] = self.score_texts(df["title"].tolist())
        daily = df.groupby("date", as_index=False)["sentiment_score"].mean()
        daily.rename(columns={"sentiment_score": "sentiment"}, inplace=True)

        # Save results for reproducibility
        os.makedirs("results_task7", exist_ok=True)
        out_path = os.path.join("results_task7", "sentiment_daily.csv")
        daily.to_csv(out_path, index=False)
        print(f"Saved daily sentiment scores → {out_path}")

        return daily

# ------------------ Utility ------------------

def merge_with_prices(df_prices: pd.DataFrame, df_sent: pd.DataFrame) -> pd.DataFrame:
    """
    Join sentiment (daily) with price data (df_prices index=Date).
    Forward-fills sentiment for trading days with no headlines.
    """
    df = df_prices.copy()
    df.index = pd.to_datetime(df.index)
    sent = df_sent.copy()
    sent["date"] = pd.to_datetime(sent["date"])
    merged = df.merge(sent, how="left", left_index=True, right_on="date").drop(columns="date")
    merged["sentiment"] = merged["sentiment"].ffill().bfill().fillna(0)
    return merged
