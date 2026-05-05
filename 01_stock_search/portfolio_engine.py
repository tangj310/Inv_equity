#!/usr/bin/env python3
"""
portfolio_engine.py
-------------------
Standalone engine that reads raw investment data sources and writes
portfolio_returns.json for the FastAPI home page.

Completely decoupled from inv_summary_02.ipynb — reproduces the same two
output tables from the raw files:
  • inv_yrly_merged_consolidate
  • consolidated_holding_cost_df_latest  (sorted by position_%)

Configuration (via .env in project root):
    alpha_vantage_api_key = <your key>
    local_inv_directry    = D:\\Investment

Usage:
    python 01_stock_search/portfolio_engine.py   # from project root
    python portfolio_engine.py                   # from inside 01_stock_search/
"""

import csv
import json
import math
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
load_dotenv()
_API_KEY  = os.getenv("alpha_vantage_api_key", "")
_DATA_DIR = Path(os.getenv("local_inv_directry", r"D:\Investment"))
_HERE     = Path(__file__).parent.resolve()
_OUTPUT   = _HERE / "portfolio_returns.json"

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.options.mode.copy_on_write = True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Market data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _av_get_json(url, max_retries=4):
    backoff = 1.0
    for _ in range(max_retries):
        time.sleep(0.25)
        data = requests.get(url, timeout=30).json()
        info = data.get("Information", "")
        if "Burst pattern" in info or "rate limit" in info.lower():
            print(f"    [throttled] sleeping {backoff:.1f}s …")
            time.sleep(backoff)
            backoff *= 2
            continue
        return data
    raise RuntimeError(f"Alpha Vantage burst limit not clearing: {data}")


def _fetch_fx(api_key):
    """Return (monthly_df, yearly_df, monthly_lookup_dict, cadusd_for_date fn)."""
    url  = (f"https://www.alphavantage.co/query?function=FX_MONTHLY"
            f"&from_symbol=CAD&to_symbol=USD&apikey={api_key}")
    data = requests.get(url, timeout=30).json()
    key  = "Time Series FX (Monthly)"
    if key not in data:
        raise ValueError(f"No FX data: {data}")

    monthly_df = (
        pd.DataFrame(data[key]).transpose()[["4. close"]]
        .rename(columns={"4. close": "CADUSD"})
    )
    monthly_df["CADUSD"] = pd.to_numeric(monthly_df["CADUSD"], errors="coerce")
    monthly_df.index     = pd.to_datetime(monthly_df.index)
    monthly_df["Year"]   = monthly_df.index.year
    monthly_df["Month"]  = monthly_df.index.month

    yearly_df = monthly_df.groupby("Year")["CADUSD"].mean().to_frame(name="CADUSD")
    yearly_df["Year"] = yearly_df.index

    monthly_df = monthly_df.sort_index(ascending=False)
    yearly_df  = yearly_df.sort_index(ascending=False)

    lookup = {
        (int(r["Year"]), int(r["Month"])): float(r["CADUSD"])
        for _, r in monthly_df.iterrows()
    }

    def cadusd_for_date(date):
        return lookup.get(
            (int(date.year), int(date.month)),
            float(yearly_df.loc[int(date.year), "CADUSD"]),
        )

    return monthly_df, yearly_df, lookup, cadusd_for_date


def _fetch_spy(api_key):
    """Return sp500_hist_annual DataFrame indexed by year."""
    url  = (f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
            f"&symbol=SPY&apikey={api_key}&outputsize=full")
    data = requests.get(url, timeout=30).json()
    if "Time Series (Daily)" not in data:
        raise ValueError(f"No SPY data: {data}")

    df = (
        pd.DataFrame(data["Time Series (Daily)"]).transpose()[["4. close"]]
        .rename(columns={"4. close": "stock_price"})
    )
    df["stock_price"] = df["stock_price"].astype(float).round(2)
    df.index = pd.to_datetime(df.index)

    annual = df.resample("Y").last()
    annual.index = annual.index.year
    annual["Year"] = annual.index
    annual["Annual Return"] = annual["stock_price"].pct_change() * 100
    return annual.dropna(subset=["Annual Return"]).sort_index(ascending=False)


def _fetch_daily_split_adjusted(symbol, api_key):
    """Return (Daily_stock_df, split_df) split-adjusted."""
    splits = _av_get_json(
        f"https://www.alphavantage.co/query?function=SPLITS&symbol={symbol}&apikey={api_key}"
    ).get("data", [])
    if splits:
        split_df = pd.DataFrame(splits)
        split_df["split_factor"]   = pd.to_numeric(split_df["split_factor"], errors="coerce")
        split_df["effective_date"] = pd.to_datetime(split_df["effective_date"])
    else:
        split_df = pd.DataFrame({"split_factor": [1], "effective_date": [datetime.today()]})

    series = _av_get_json(
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={symbol}&apikey={api_key}&outputsize=full"
    ).get("Time Series (Daily)")
    if series is None:
        raise ValueError(f"No daily data for {symbol}")

    daily_df = (
        pd.DataFrame(series).transpose()[["4. close"]]
        .rename(columns={"4. close": "stock_price"})
    )
    daily_df["stock_price"] = daily_df["stock_price"].astype(float).round(2)
    daily_df.index = pd.to_datetime(daily_df.index)

    for eff_date in split_df["effective_date"].dt.date:
        factor = split_df.loc[split_df["effective_date"].dt.date == eff_date, "split_factor"].values[0]
        daily_df.loc[daily_df.index.date < eff_date, "stock_price"] /= factor

    return daily_df, split_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Questrade
# ══════════════════════════════════════════════════════════════════════════════

def _load_questrade(data_dir, api_key, fx_yrly_df, fx_monthly_lookup, cadusd_for_date):
    """Return (inv_yrly_base, questrade_cf_df, trades_df_questrade)."""
    files = [
        data_dir / "questrade_Inv_activity_2019-2021.xlsx",
        data_dir / "questrade_Inv_activity_2022.xlsx",
        data_dir / "questrade_Inv_activity_2023.xlsx",
        data_dir / "questrade_Inv_activity_2024.xlsx",
        data_dir / "questrade_Inv_activity_2025.xlsx",
        data_dir / "questrade_Inv_activity_2026.xlsx",
    ]
    df = pd.concat([pd.read_excel(f) for f in files if f.exists()], ignore_index=True)
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
    df["Year"] = df["Transaction Date"].dt.year
    df = df[df["Year"] >= 2019]
    df.rename(columns={
        "Transaction Date": "Transaction_Date",
        "Settlement Date":  "Settlement_Date",
        "Net Amount":       "Net_Amount",
    }, inplace=True)

    # ── Deposits / Withdrawals / Dividends ────────────────────────────────
    ddw_categories = {
        "Deposits":    ["CON", "FCH", "DEP"],
        "Withdrawals": ["WDR"],
        "Dividends":   ["DIV", "Dividends"],
    }
    yearly_totals = {}
    for cat, actions in ddw_categories.items():
        sub = df[df["Action"].isin(actions) | df["Activity Type"].isin(actions)].copy()
        _cad = sub["Currency"] == "CAD"
        sub.loc[_cad, "Net_Amount"] = sub.loc[_cad].apply(
            lambda row: row["Net_Amount"] * cadusd_for_date(row["Transaction_Date"]), axis=1
        )
        yearly_totals[cat] = sub.groupby("Year")["Net_Amount"].sum()

    inv_yrly_base = pd.DataFrame(yearly_totals).fillna(0)

    _HARDCODED_2019_CAD = 6000.0
    inv_yrly_base.at[2019, "Deposits"] = (
        _HARDCODED_2019_CAD * cadusd_for_date(pd.Timestamp("2019-01-02"))
    )

    # ── Per-transaction cash flows (for Modified Dietz / XIRR) ───────────
    cf_actions = ["CON", "FCH", "DEP", "WDR"]
    cf_df = df[df["Action"].isin(cf_actions) | df["Activity Type"].isin(cf_actions)][
        ["Transaction_Date", "Net_Amount", "Currency", "Year"]
    ].copy()
    _cad = cf_df["Currency"] == "CAD"
    cf_df.loc[_cad, "Net_Amount"] = cf_df.loc[_cad].apply(
        lambda row: row["Net_Amount"] * cadusd_for_date(row["Transaction_Date"]), axis=1
    )
    cf_df = cf_df[["Transaction_Date", "Net_Amount"]].reset_index(drop=True)

    # Reconcile 2019 synthetic deposit
    target_2019 = _HARDCODED_2019_CAD * cadusd_for_date(pd.Timestamp("2019-01-02"))
    actual_2019 = cf_df.loc[cf_df["Transaction_Date"].dt.year == 2019, "Net_Amount"].sum()
    missing     = target_2019 - actual_2019
    if missing > 0:
        cf_df = pd.concat([
            cf_df,
            pd.DataFrame([{"Transaction_Date": pd.Timestamp("2019-01-02"), "Net_Amount": missing}]),
        ], ignore_index=True)
        print(f"  2019 reconcile: added synthetic ${missing:,.2f} USD")

    # ── Trades ────────────────────────────────────────────────────────────
    exclude = {"DLR.TO", "G036247", "H038778", "DLR", "IVV", "VYM", "VFH", "KWEB", "BYND"}
    trades = df[
        (df["Activity Type"] == "Trades") & (~df["Symbol"].isin(exclude))
    ].copy()

    if datetime.today().year not in sorted(trades["Year"].unique()):
        trades = trades.sort_values("Transaction_Date").reset_index(drop=True)
        last   = trades.iloc[-1]
        cy     = datetime.now().year
        buy    = pd.DataFrame([last])
        sell   = pd.DataFrame([last])
        buy["Action"] = "Buy"; buy["Transaction_Date"] = pd.Timestamp(f"{cy}-01-01")
        buy["Settlement_Date"] = pd.Timestamp(f"{cy}-01-01"); buy["Year"] = cy
        buy["Quantity"] = abs(last["Quantity"])
        buy["Net_Amount"] = abs(last["Quantity"]) * last["Price"] * -1
        sell["Action"] = "Sell"; sell["Transaction_Date"] = pd.Timestamp(f"{cy}-01-02")
        sell["Settlement_Date"] = pd.Timestamp(f"{cy}-01-02"); sell["Year"] = cy
        sell["Quantity"] = abs(last["Quantity"]) * -1
        sell["Net_Amount"] = abs(last["Quantity"]) * last["Price"]
        trades = pd.concat([trades, buy, sell], ignore_index=True)
        print("  Questrade: created synthetic current-year trades")

    _cad = trades["Currency"] == "CAD"
    trades.loc[_cad, "Net_Amount"] = trades.loc[_cad].apply(
        lambda row: row["Net_Amount"] * cadusd_for_date(row["Transaction_Date"]), axis=1
    )
    trades.loc[_cad, "Price"] = trades.loc[_cad].apply(
        lambda row: row["Price"] * cadusd_for_date(row["Transaction_Date"]), axis=1
    )
    trades = trades[["Transaction_Date", "Action", "Symbol", "Quantity", "Price", "Net_Amount", "Year"]]

    # ── Open positions per year ────────────────────────────────────────────
    open_prev = pd.DataFrame()
    for i in sorted(trades["Year"].unique()):
        open_mkt_vals = []
        yr_trades = trades[trades["Year"] == i]
        inv_yrly_base.loc[inv_yrly_base.index == i, "Closed_gain/loss"] = yr_trades["Net_Amount"].sum()

        yr_trades = pd.concat([yr_trades, open_prev], ignore_index=True)
        buy_df  = yr_trades[yr_trades["Action"] == "Buy"]
        sell_df = yr_trades[yr_trades["Action"] == "Sell"]

        buy_grp  = buy_df.groupby("Symbol")["Quantity"].sum()
        sell_grp = sell_df.groupby("Symbol")["Quantity"].sum().reindex(buy_grp.index, fill_value=0)
        open_df  = (buy_grp + sell_grp)[(buy_grp + sell_grp) > 0].reset_index()

        open_buy = pd.merge(open_df, buy_df, on="Symbol", how="left")
        open_buy = open_buy.drop_duplicates(subset=["Symbol", "Quantity_x"], keep="first")
        open_buy = open_buy[[
            "Transaction_Date", "Action", "Symbol", "Quantity_x", "Price", "Net_Amount", "Year"
        ]].rename(columns={"Quantity_x": "Quantity"})
        open_prev = open_buy.copy()

        print(f"  QT {i}: {list(open_buy['Symbol'].unique())}")
        for sym in open_buy["Symbol"].unique():
            daily_df, _ = _fetch_daily_split_adjusted(sym, api_key)
            yr_df       = daily_df.resample("Y").last()
            if sym.endswith(".TO"):
                raw_price = yr_df.loc[yr_df.index.year == i, "stock_price"].values[0]
                dec_fx    = fx_monthly_lookup.get(
                    (int(i), 12),
                    fx_monthly_lookup.get((int(i), 11), float(fx_yrly_df.loc[int(i), "CADUSD"]))
                )
                price = raw_price * dec_fx
            else:
                price = yr_df.loc[yr_df.index.year == i, "stock_price"].values[0]
            qty = open_buy.loc[open_buy["Symbol"] == sym, "Quantity"].values.sum()
            open_mkt_vals.append(qty * price)

        inv_yrly_base.loc[inv_yrly_base.index == i, "Open_mkt_value"] = round(sum(open_mkt_vals), 2)

    return inv_yrly_base, cf_df, trades


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Moomoo
# ══════════════════════════════════════════════════════════════════════════════

def _load_moomoo(data_dir, api_key, cadusd_for_date):
    """Return (inv_yrly_base_moomoo, moomoo_cf_df, trades_df_moomoo)."""
    files = [
        data_dir / "moomoo_inv_activity_2024.csv",
        data_dir / "moomoo_inv_activity_2025.csv",
        data_dir / "moomoo_inv_activity_2026.csv",
    ]
    raw = pd.concat([pd.read_csv(f) for f in files if f.exists()], ignore_index=True)
    raw["ReportDate"] = pd.to_datetime(raw["ReportDate"], format="%Y%m%d")
    raw["Year"]       = raw["ReportDate"].dt.year

    # ── Deposits / Withdrawals / Dividends ────────────────────────────────
    # Filter: (CAD & DEP) | DIV | WITH  — preserves original precedence
    ddw = raw[
        ((raw["CurrencyPrimary"] == "CAD") & (raw["ActivityCode"] == "DEP"))
        | raw["ActivityCode"].isin(["DIV", "WITH"])
    ][["ReportDate", "ActivityCode", "Amount", "CurrencyPrimary", "TransactionID", "Year"]].copy()
    ddw = ddw.sort_values("ReportDate").drop_duplicates(["TransactionID"])

    ddw_categories = {"Deposits": ["DEP"], "Withdrawals": ["WITH"], "Dividends": ["DIV"]}
    yearly_totals  = {}
    for cat, actions in ddw_categories.items():
        sub  = ddw[ddw["ActivityCode"].isin(actions)].copy()
        _cad = sub["CurrencyPrimary"] == "CAD"
        sub.loc[_cad, "Amount"] = sub.loc[_cad].apply(
            lambda row: row["Amount"] * cadusd_for_date(row["ReportDate"]), axis=1
        )
        yearly_totals[cat] = sub.groupby("Year")["Amount"].sum()

    inv_yrly_base_moomoo = pd.DataFrame(yearly_totals).fillna(0)

    # Cash flows (deposits + withdrawals only)
    cf_raw  = ddw[ddw["ActivityCode"].isin(["DEP", "WITH"])][["ReportDate", "Amount", "CurrencyPrimary"]].copy()
    _cad    = cf_raw["CurrencyPrimary"] == "CAD"
    cf_raw.loc[_cad, "Amount"] = cf_raw.loc[_cad].apply(
        lambda row: row["Amount"] * cadusd_for_date(row["ReportDate"]), axis=1
    )
    cf_df = (
        cf_raw.rename(columns={"ReportDate": "Transaction_Date", "Amount": "Net_Amount"})
        [["Transaction_Date", "Net_Amount"]].reset_index(drop=True)
    )

    # ── Trades ────────────────────────────────────────────────────────────
    trades = raw[
        raw["ActivityCode"].isin(["BUY", "SELL"])
        & raw["AssetClass"].isin(["CASH", "STK"])
        & (raw["CurrencyPrimary"] == "USD")
    ][["ReportDate", "ActivityCode", "Buy/Sell", "TradeQuantity", "TradeGross",
       "Amount", "CurrencyPrimary", "AssetClass", "Symbol", "Year"]].copy()

    if datetime.today().year not in sorted(trades["Year"].unique()):
        trades = trades.sort_values("ReportDate").reset_index(drop=True)
        last   = trades.iloc[-1]
        cy     = datetime.now().year
        buy    = pd.DataFrame([last]); sell = pd.DataFrame([last])
        buy["ActivityCode"]  = "BUY";  buy["Buy/Sell"]  = "Buy"
        buy["ReportDate"]    = pd.Timestamp(f"{cy}-01-01"); buy["Year"] = cy
        buy["TradeQuantity"] = abs(last["TradeQuantity"])
        buy["Amount"]        = abs(last["Amount"]) * -1
        sell["ActivityCode"] = "SELL"; sell["Buy/Sell"] = "Sell"
        sell["ReportDate"]   = pd.Timestamp(f"{cy}-01-02"); sell["Year"] = cy
        sell["TradeQuantity"] = abs(last["TradeQuantity"]) * -1
        sell["Amount"]        = abs(last["Amount"])
        trades = pd.concat([trades, buy, sell], ignore_index=True)
        print("  Moomoo: created synthetic current-year trades")

    trades = trades.rename(columns={
        "ReportDate": "Transaction_Date", "ActivityCode": "Action",
        "TradeQuantity": "Quantity", "Amount": "Net_Amount",
    })
    trades["Action"] = trades["Action"].replace({"BUY": "Buy", "SELL": "Sell"})
    trades["Price"]  = trades["TradeGross"] / trades["Quantity"]
    trades = trades[["Transaction_Date", "Action", "Symbol", "Quantity", "Price", "Net_Amount", "Year"]]

    # ── Open positions per year ────────────────────────────────────────────
    open_prev = pd.DataFrame()
    for i in sorted(trades["Year"].unique()):
        open_mkt_vals = []
        yr_trades = trades[trades["Year"] == i]
        inv_yrly_base_moomoo.loc[inv_yrly_base_moomoo.index == i, "Closed_gain/loss"] = (
            yr_trades["Net_Amount"].sum()
        )

        yr_trades = pd.concat([yr_trades, open_prev], ignore_index=True)
        buy_df    = yr_trades[yr_trades["Action"] == "Buy"]
        sell_df   = yr_trades[yr_trades["Action"] == "Sell"]

        buy_grp  = buy_df.groupby("Symbol")["Quantity"].sum()
        sell_grp = sell_df.groupby("Symbol")["Quantity"].sum().reindex(buy_grp.index, fill_value=0)
        open_df  = (buy_grp + sell_grp)[(buy_grp + sell_grp) > 0].reset_index()

        open_buy = pd.merge(open_df, buy_df, on="Symbol", how="left")
        open_buy = open_buy.drop_duplicates(subset=["Symbol", "Quantity_x"], keep="first")
        open_buy = open_buy[[
            "Transaction_Date", "Action", "Symbol", "Quantity_x", "Price", "Net_Amount", "Year"
        ]].rename(columns={"Quantity_x": "Quantity"})
        open_prev = open_buy.copy()

        print(f"  MM {i}: {list(open_buy['Symbol'].unique())}")
        for sym in open_buy["Symbol"].unique():
            daily_df, _ = _fetch_daily_split_adjusted(sym, api_key)
            yr_df        = daily_df.resample("Y").last()
            price        = yr_df.loc[yr_df.index.year == i, "stock_price"].values[0]
            qty          = open_buy.loc[open_buy["Symbol"] == sym, "Quantity"].values.sum()
            open_mkt_vals.append(qty * price)

        inv_yrly_base_moomoo.loc[inv_yrly_base_moomoo.index == i, "Open_mkt_value"] = round(
            sum(open_mkt_vals), 2
        )

    return inv_yrly_base_moomoo, cf_df, trades


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Consolidation + Return calculations
# ══════════════════════════════════════════════════════════════════════════════

def _modified_dietz_twr(year, v_start, v_end, cf_df, today):
    year_start  = pd.Timestamp(f"{int(year)}-01-01")
    period_end  = min(pd.Timestamp(f"{int(year)}-12-31"), today)
    period_days = (period_end - year_start).days
    if period_days <= 0:
        return float("nan")
    in_yr = cf_df[
        (cf_df["Transaction_Date"] >= year_start)
        & (cf_df["Transaction_Date"] <= period_end)
    ]
    if in_yr.empty:
        net_cf = 0.0; weighted_cf = 0.0
    else:
        net_cf      = float(in_yr["Net_Amount"].sum())
        days_after  = (period_end - in_yr["Transaction_Date"]).dt.days
        weighted_cf = float((days_after / period_days * in_yr["Net_Amount"]).sum())
    numer = v_end - v_start - net_cf
    denom = v_start + weighted_cf
    return numer / denom if denom != 0 else float("nan")


def _xirr(dates, amounts, lo=-0.99, hi=10.0, tol=1e-9, max_iter=200):
    if not dates or len(dates) != len(amounts):
        return None
    pairs    = sorted(zip(dates, amounts), key=lambda p: p[0])
    dates_s, amounts_s = zip(*pairs)
    t0   = dates_s[0]
    days = [(d - t0).days for d in dates_s]
    def npv(r): return sum(a / (1 + r) ** (d / 365.25) for d, a in zip(days, amounts_s))
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi); f_mid = npv(mid)
        if abs(f_mid) < tol: return mid
        if f_lo * f_mid < 0: hi, f_hi = mid, f_mid
        else:                 lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def _compute_returns(inv_yrly_base, inv_yrly_base_moomoo, trades_merge,
                     qt_cf, mm_cf, sp500_hist_annual, fx_yrly_df):
    """Return (inv_yrly_merged, inv_yrly_merged_consolidate)."""

    # Merge brokerage tables
    base = inv_yrly_base.reset_index(drop=False)
    mm   = inv_yrly_base_moomoo.reset_index(drop=False)
    merged = (
        pd.concat([base, mm]).groupby("Year").sum().reset_index(drop=False).fillna(0)
    )

    cadusd_map = dict(zip(fx_yrly_df["Year"], fx_yrly_df["CADUSD"]))
    merged["CADUSD_forx"] = merged["Year"].map(cadusd_map)

    # Cumulative columns
    merged["Activity_Yr"]         = merged["Year"]
    merged["Principal"]           = merged["Deposits"] + merged["Withdrawals"]
    merged["Principal_after_div"] = merged["Principal"] + merged["Dividends"]
    merged["Running_Principal"]           = merged["Principal"].cumsum()
    merged["Running_Dividends"]           = merged["Dividends"].cumsum()
    merged["Running_Closed_gain/loss"]    = merged["Closed_gain/loss"].cumsum()
    merged["Running_Principal_after_div"] = merged["Principal_after_div"].cumsum()
    merged["Year_end_capital_value"] = (
        merged["Running_Principal_after_div"]
        + merged["Running_Closed_gain/loss"]
        + merged["Open_mkt_value"]
    )

    # Realized P&L via avg-cost ledger (reference column)
    _t = trades_merge.copy()
    _t["_sort_key"] = _t["Action"].map({"Buy": 0, "Sell": 1})
    _t = _t.sort_values(["Transaction_Date", "_sort_key"]).reset_index(drop=True)
    pnl_by_year = {int(y): 0.0 for y in merged["Year"].unique()}
    inv = {}
    for _, row in _t.iterrows():
        sym = row["Symbol"]; yr = int(row["Year"])
        if sym not in inv: inv[sym] = {"qty": 0.0, "cost": 0.0}
        if row["Action"] == "Buy":
            inv[sym]["qty"]  += row["Quantity"]
            inv[sym]["cost"] += -row["Net_Amount"]
        elif row["Action"] == "Sell":
            sq = abs(row["Quantity"])
            if inv[sym]["qty"] > 0:
                avg = inv[sym]["cost"] / inv[sym]["qty"]
                pnl_by_year[yr] = pnl_by_year.get(yr, 0.0) + (row["Net_Amount"] - avg * sq)
                inv[sym]["qty"]  -= sq
                inv[sym]["cost"] -= avg * sq
    merged["Realized_PnL"] = merged["Year"].map(pnl_by_year).fillna(0)

    # Combined cash flow series
    cf_df = pd.concat([qt_cf, mm_cf], ignore_index=True)
    cf_df["Transaction_Date"] = pd.to_datetime(cf_df["Transaction_Date"])
    cf_df = cf_df.sort_values("Transaction_Date").reset_index(drop=True)
    today = pd.Timestamp.today().normalize()

    merged = merged.sort_values("Year").reset_index(drop=True)
    merged["Year_start_capital_value"] = merged["Year_end_capital_value"].shift(1).fillna(0)

    # Modified Dietz TWR
    merged["Ratio_Yearly_Return_%"] = [
        _modified_dietz_twr(
            row["Year"], row["Year_start_capital_value"],
            row["Year_end_capital_value"], cf_df, today,
        ) * 100
        for _, row in merged.iterrows()
    ]
    merged["Ratio_Dividends_Yield_%"] = (
        merged["Dividends"] / merged["Running_Principal_after_div"]
    ) * 100

    # CAGR (chain-linked TWR)
    start_date    = pd.Timestamp(f"{int(merged['Year'].min())}-01-01")
    n_years       = (today - start_date).days / 365.25
    hpr           = (1 + merged["Ratio_Yearly_Return_%"] / 100).prod()
    merged["CAGR_%"] = (hpr ** (1 / n_years) - 1) * 100

    sp_end   = sp500_hist_annual[sp500_hist_annual["Year"] == today.year]["stock_price"].values.round(2)[0]
    sp_start = sp500_hist_annual[sp500_hist_annual["Year"] == int(merged["Year"].min()) - 1]["stock_price"].values.round(2)[0]
    merged["sp500_CAGR_%"] = ((sp_end / sp_start) ** (1 / n_years) - 1) * 100

    # XIRR / MWR
    xirr_dates   = list(cf_df["Transaction_Date"])
    xirr_amounts = [-x for x in cf_df["Net_Amount"].tolist()]
    xirr_dates.append(today)
    xirr_amounts.append(merged["Year_end_capital_value"].iloc[-1])
    irr = _xirr(xirr_dates, xirr_amounts)
    merged["MWR_IRR_%"] = (irr * 100) if irr is not None else float("nan")
    merged.reset_index(drop=True, inplace=True)

    # Join S&P 500 annual returns
    inv_yrly_merged = merged.merge(
        sp500_hist_annual, how="left", left_on="Activity_Yr", right_on="Year"
    )
    inv_yrly_merged.rename(columns={"Annual Return": "sp500_without_div_%"}, inplace=True)

    display_cols = ["Activity_Yr", "Ratio_Yearly_Return_%", "sp500_without_div_%",
                    "CAGR_%", "sp500_CAGR_%", "MWR_IRR_%"]
    round_cols   = ["Ratio_Yearly_Return_%", "sp500_without_div_%", "CAGR_%", "sp500_CAGR_%", "MWR_IRR_%"]
    inv_yrly_merged[round_cols] = inv_yrly_merged[round_cols].round(2)
    inv_yrly_merged_consolidate = inv_yrly_merged[display_cols]

    return inv_yrly_merged, inv_yrly_merged_consolidate


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Current holdings
# ══════════════════════════════════════════════════════════════════════════════

def _compute_holdings(trades_merge, inv_yrly_merged, api_key):
    """Return consolidated_holding_cost_df_latest sorted by position_%."""

    # Build year-end open stock dictionary
    open_prev = pd.DataFrame()
    year_end_dict = {}

    for i in sorted(trades_merge["Year"].unique()):
        year_end_dict[i] = {"Symbol": [], "Quantity": []}
        yr_trades = pd.concat(
            [trades_merge[trades_merge["Year"] == i], open_prev], ignore_index=True
        )
        buy_df  = yr_trades[yr_trades["Action"] == "Buy"]
        sell_df = yr_trades[yr_trades["Action"] == "Sell"]

        buy_grp  = buy_df.groupby("Symbol")["Quantity"].sum()
        sell_grp = sell_df.groupby("Symbol")["Quantity"].sum().reindex(buy_grp.index, fill_value=0)
        open_df  = (buy_grp + sell_grp)[(buy_grp + sell_grp) > 0].reset_index()

        open_buy = pd.merge(open_df, buy_df, on="Symbol", how="left")
        open_buy = open_buy.drop_duplicates(subset=["Symbol", "Quantity_x"], keep="first")
        open_buy = open_buy[[
            "Transaction_Date", "Action", "Symbol", "Quantity_x", "Price", "Net_Amount", "Year"
        ]].rename(columns={"Quantity_x": "Quantity"})
        open_prev = open_buy.copy()

        for sym in open_buy["Symbol"].unique():
            qty = open_buy[open_buy["Symbol"] == sym]["Quantity"].values.sum()
            year_end_dict[i]["Symbol"].append(sym)
            year_end_dict[i]["Quantity"].append(qty)

    # Current holdings transaction history
    cur_yr   = datetime.today().year
    cur_syms = set(year_end_dict[cur_yr]["Symbol"])
    holding_df = pd.concat([
        trades_merge[trades_merge["Symbol"] == sym] for sym in cur_syms
    ]).drop_duplicates().sort_values("Transaction_Date", ascending=True)

    # Rolling cost basis per symbol
    cost_dict = {}
    for sym in holding_df["Symbol"].unique():
        df = holding_df[holding_df["Symbol"] == sym].copy()
        df["Rolling_Capital"]         = df["Net_Amount"].cumsum()
        df["Rolling_Quantity"]        = df["Quantity"].cumsum()
        df["Rolling_Price_Per_Share"] = df["Rolling_Capital"] / df["Rolling_Quantity"]
        cost_dict[sym] = df[[
            "Transaction_Date", "Symbol", "Action", "Quantity",
            "Rolling_Quantity", "Price", "Rolling_Capital", "Rolling_Price_Per_Share",
        ]]

    consolidated = pd.concat(cost_dict.values())
    dupe_groups  = consolidated.groupby(["Symbol", "Transaction_Date"]).cumcount()
    consolidated["Transaction_Date"] += pd.to_timedelta(dupe_groups, unit="d")

    latest = (
        consolidated
        .loc[consolidated.groupby("Symbol")["Transaction_Date"].idxmax()]
        .reset_index(drop=True)
    )
    latest["position_%"] = np.nan
    total_value = inv_yrly_merged["Year_end_capital_value"].values[-1]

    # Fetch current prices + position% + earnings
    print("\nFetching current prices and earnings …")
    latest["Next_qtr_date"] = None

    for sym in latest["Symbol"].unique():
        # Current price
        daily_df, _ = _fetch_daily_split_adjusted(sym, api_key)
        latest.loc[latest["Symbol"] == sym, "latest_stock_price"] = daily_df["stock_price"].iloc[0]

        # Cost-based position%
        rc = latest.loc[latest["Symbol"] == sym, "Rolling_Capital"].values
        holding_cost = abs(rc) if rc < 0 else 1
        latest.loc[latest["Symbol"] == sym, "position_%"] = (
            (holding_cost / total_value) * 100
        ).round(2)
        latest.loc[latest["Symbol"] == sym, "Year_end_capital_value"] = total_value

        # Next earnings date
        csv_url = (f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR"
                   f"&symbol={sym}&horizon=12month&apikey={api_key}")
        with requests.Session() as s:
            decoded = s.get(csv_url, timeout=30).content.decode("utf-8")
            rows    = list(csv.reader(decoded.splitlines(), delimiter=","))
            if len(rows) > 1:
                fc = pd.DataFrame(columns=rows[0], data=rows[1:])
                if "reportDate" in fc.columns and not fc.empty:
                    latest.loc[latest["Symbol"] == sym, "Next_qtr_date"] = (
                        fc["reportDate"].head(1).values[0]
                    )

    latest = latest[[
        "Transaction_Date", "Symbol", "Rolling_Quantity", "Rolling_Capital",
        "Rolling_Price_Per_Share", "position_%", "Year_end_capital_value",
        "latest_stock_price", "Next_qtr_date",
    ]]

    return latest.sort_values(by="position_%", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — JSON serialisation + main
# ══════════════════════════════════════════════════════════════════════════════

def _safe(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _returns_to_payload(consolidate_df, full_merged_df):
    first = consolidate_df.iloc[0]
    cagr  = _safe(first.get("CAGR_%"))
    sp500c = _safe(first.get("sp500_CAGR_%"))
    mwr   = _safe(first.get("MWR_IRR_%"))

    rows = []
    for _, r in consolidate_df.iterrows():
        rows.append({
            "Activity_Yr":           int(r["Activity_Yr"]),
            "Ratio_Yearly_Return_%": _safe(r.get("Ratio_Yearly_Return_%")),
            "sp500_without_div_%":   _safe(r.get("sp500_without_div_%")),
        })
    return cagr, sp500c, mwr, rows


def _holdings_to_payload(holdings_df):
    result = []
    for _, r in holdings_df.iterrows():
        result.append({
            "Symbol":                  r.get("Symbol"),
            "Rolling_Quantity":        _safe(r.get("Rolling_Quantity")),
            "Rolling_Capital":         _safe(r.get("Rolling_Capital")),
            "Rolling_Price_Per_Share": _safe(r.get("Rolling_Price_Per_Share")),
            "position_%":              _safe(r.get("position_%")),
            "latest_stock_price":      _safe(r.get("latest_stock_price")),
            "Next_qtr_date":           str(r.get("Next_qtr_date")) if r.get("Next_qtr_date") else None,
        })
    return result


def main():
    if not _API_KEY:
        raise EnvironmentError("alpha_vantage_api_key not set in .env")
    if not _DATA_DIR.exists():
        raise FileNotFoundError(f"Investment data directory not found: {_DATA_DIR}")

    print("=" * 60)
    print("Portfolio Engine")
    print(f"Data dir : {_DATA_DIR}")
    print(f"Output   : {_OUTPUT}")
    print("=" * 60)

    print("\n[1/5] Fetching FX and SPY data …")
    fx_monthly_df, fx_yrly_df, fx_monthly_lookup, cadusd_for_date = _fetch_fx(_API_KEY)
    sp500_hist_annual = _fetch_spy(_API_KEY)
    print(f"  FX: {fx_monthly_df.index[-1].date()} – {fx_monthly_df.index[0].date()}")
    print(f"  SPY annual rows: {len(sp500_hist_annual)}")

    print("\n[2/5] Loading Questrade data …")
    inv_yrly_base, qt_cf, trades_qt = _load_questrade(
        _DATA_DIR, _API_KEY, fx_yrly_df, fx_monthly_lookup, cadusd_for_date
    )

    print("\n[3/5] Loading Moomoo data …")
    inv_yrly_base_moomoo, mm_cf, trades_mm = _load_moomoo(
        _DATA_DIR, _API_KEY, cadusd_for_date
    )

    trades_merge = pd.concat([trades_qt, trades_mm], ignore_index=True)

    print("\n[4/5] Computing returns …")
    inv_yrly_merged, inv_yrly_merged_consolidate = _compute_returns(
        inv_yrly_base, inv_yrly_base_moomoo, trades_merge,
        qt_cf, mm_cf, sp500_hist_annual, fx_yrly_df,
    )
    print(inv_yrly_merged_consolidate.to_string(index=False))

    print("\n[5/5] Computing current holdings …")
    holdings_df = _compute_holdings(trades_merge, inv_yrly_merged, _API_KEY)
    print(holdings_df[["Symbol", "Rolling_Quantity", "position_%", "latest_stock_price"]].to_string(index=False))

    # Write JSON
    cagr, sp500c, mwr, return_rows = _returns_to_payload(
        inv_yrly_merged_consolidate, inv_yrly_merged
    )
    holding_rows = _holdings_to_payload(holdings_df)

    payload = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "CAGR_%":       cagr,
        "sp500_CAGR_%": sp500c,
        "MWR_IRR_%":    mwr,
        "rows":         return_rows,
        "holdings":     holding_rows,
    }
    with open(_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWritten: {_OUTPUT}")
    print(f"  Returns:  {len(return_rows)} rows")
    print(f"  Holdings: {len(holding_rows)} positions")
    print(f"  CAGR: {cagr}%  |  S&P 500 CAGR: {sp500c}%  |  MWR/IRR: {mwr}%")


if __name__ == "__main__":
    main()
