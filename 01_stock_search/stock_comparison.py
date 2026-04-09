import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from cache import cache_get, cache_set


def process_ticker_return(symbol: str, years: float, api_key: str) -> dict:
    """
    Fetch split-adjusted daily prices + dividends for one ticker,
    filter to the requested year window, normalize so the start of
    the window = 1.0, and compute the total-return multiplier
    (price + dividends) / start_price.

    Returns:
        {
          ticker, dates, values,          # normalised price series
          return_multiplier,              # total return including dividends
          price_return_multiplier,        # price-only return
          start_price, end_price,
          total_dividends
        }
    Raises on fatal error.
    """
    cache_key = f"return:{symbol}:{years}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    today      = datetime.today()
    start_date = today - timedelta(days=int(years * 365))   # oldest boundary
    end_date   = today                                       # newest boundary

    # ── Split factor ─────────────────────────────────────────────────────
    url  = f"https://www.alphavantage.co/query?function=SPLITS&symbol={symbol}&apikey={api_key}"
    data = requests.get(url).json()

    split_df = pd.DataFrame({"split_factor": [1], "effective_date": [today]})
    for key, value in data.items():
        if key == "data" and len(value) > 0:
            split_df = pd.DataFrame(value)
            split_df["split_factor"]  = pd.to_numeric(split_df["split_factor"], errors="coerce")
            split_df["effective_date"] = pd.to_datetime(split_df["effective_date"])

    # ── Daily prices (full history) ───────────────────────────────────────
    url  = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full"
    data = requests.get(url).json()

    daily_df = pd.DataFrame()
    for key, value in data.items():
        if key == "Time Series (Daily)":
            daily_df = pd.DataFrame(value).T[["4. close"]].rename(columns={"4. close": "price"})
            daily_df["price"] = daily_df["price"].astype(float).round(4)
            daily_df.index    = pd.to_datetime(daily_df.index)

    if daily_df.empty:
        raise ValueError("No daily price data returned from Alpha Vantage")

    # Adjust for splits
    for date_i in daily_df.index.date:
        for date_j in split_df["effective_date"].dt.date:
            if date_i == date_j:
                factor = split_df.loc[split_df["effective_date"].dt.date == date_j, "split_factor"].values[0]
                daily_df.loc[daily_df.index.date < date_j, "price"] /= factor

    # Sort ascending (oldest first), filter to window
    daily_df = daily_df.sort_index(ascending=True)
    mask     = (daily_df.index.date >= start_date.date()) & (daily_df.index.date <= end_date.date())
    window_df = daily_df[mask].copy()

    if window_df.empty or len(window_df) < 2:
        raise ValueError(f"Not enough price data in the requested {years}-year window")

    # ── Dividends ─────────────────────────────────────────────────────────
    url  = f"https://www.alphavantage.co/query?function=DIVIDENDS&symbol={symbol}&apikey={api_key}"
    data = requests.get(url).json()

    total_dividends = 0.0
    for key, value in data.items():
        if key == "data" and len(value) > 0:
            div_df = pd.DataFrame(value)[["ex_dividend_date", "amount"]]
            div_df["ex_dividend_date"] = pd.to_datetime(div_df["ex_dividend_date"])
            div_df["amount"]           = pd.to_numeric(div_df["amount"], errors="coerce").fillna(0)
            mask_div = (
                (div_df["ex_dividend_date"] >= pd.Timestamp(start_date))
                & (div_df["ex_dividend_date"] <= pd.Timestamp(end_date))
            )
            total_dividends = float(div_df.loc[mask_div, "amount"].sum())

    # ── Normalise & compute returns ───────────────────────────────────────
    start_price = float(window_df["price"].iloc[0])   # oldest date in window
    end_price   = float(window_df["price"].iloc[-1])  # most recent date

    normalized = (window_df["price"] / start_price).round(6)

    price_return_multiplier = round(end_price / start_price, 4)
    total_return_multiplier = round((end_price + total_dividends) / start_price, 4)

    # Sanitise for JSON
    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    result = {
        "ticker":                 symbol,
        "dates":                  normalized.index.strftime("%Y-%m-%d").tolist(),
        "values":                 [_clean(v) for v in normalized.tolist()],
        "return_multiplier":      total_return_multiplier,
        "price_return_multiplier": price_return_multiplier,
        "start_price":            round(start_price, 2),
        "end_price":              round(end_price, 2),
        "total_dividends":        round(total_dividends, 2),
    }
    cache_set(cache_key, result)
    return result
