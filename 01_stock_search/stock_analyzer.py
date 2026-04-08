import csv as csv_module
import numpy as np
import pandas as pd
import requests
import warnings
import yfinance as yf
from datetime import datetime
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.options.mode.copy_on_write = True

# Columns returned for the single-stock detail view (mirrors ticker_dict_pd in the notebook)
DETAIL_COLS = [
    "Ticker",
    "stock_price",
    "EPS_currentQtr",
    "EPS_nextQtr",
    "EPS_TTM",
    "EPS_nextQtr_TTM",
    "EPS_nextYr",
    "PE_TTM",
    "PE_TTM_avg",
    "PE_TTM_volatility_+",
    "PE_TTM_volatility_-",
    # dynamic: PE_{n}yr_avg / volatility_+/-  — added at runtime
    "relative_valuation_TTM_+",
    "relative_valuation_TTM_-",
    "relative_valuation_TTM_median",
    "relative_valuation_nextQuater_projected_+",
    "relative_valuation_nextQuater_projected_-",
    "relative_valuation_nextQuater_projected_median",
    "relative_valuation_nextYear_projected_+",
    "relative_valuation_nextYear_projected_-",
    "relative_valuation_nextYear_projected_median",
    "price_valuation_assessment",
    "EPS_nextQtr_growthRate",
    "EPS_nextYr_growthRate",
    "EarningYield_TTM",
    "ERP_TTM",
    "BVPS_latest",
    "PFCF_TTM",
    "FCF_yield_TTM",
]

# Columns for the screener view (one row per ticker)
SCREEN_COLS = [
    "Ticker",
    "BVPS_latest",
    "stock_price",
    "EPS_TTM",
    "EPS_nextYr",
    "PE_TTM",
    "PE_TTM_avg",
    # dynamic PE yr cols added at runtime
    "relative_valuation_TTM_+",
    "relative_valuation_TTM_-",
    "relative_valuation_TTM_median",
    "relative_valuation_nextQuater_projected_+",
    "relative_valuation_nextQuater_projected_-",
    "relative_valuation_nextQuater_projected_median",
    "relative_valuation_nextYear_projected_+",
    "relative_valuation_nextYear_projected_-",
    "relative_valuation_nextYear_projected_median",
    "price_valuation_assessment",
    "EPS_nextQtr_growthRate",
    "EPS_nextYr_growthRate",
    "ERP_TTM",
    "PFCF_TTM",
    "FCF_yield_TTM",
    "next_yr_days7ago_EPS",
    "next_yr_days30ago_EPS",
    "next_yr_days60ago_EPS",
    "next_yr_days90ago_EPS",
    "curr_yr_growthrate_symbol",
    "next_yr_growthrate_symbol",
    "curr_yr_growthrate_index",
    "next_yr_growthrate_index",
]


def _safe_round(v, n=2):
    return round(float(v), n) if v is not None else None


def _clean_df_for_json(df: pd.DataFrame) -> list:
    """Replace inf/NaN with None and return list of dicts."""
    clean = df.replace([np.inf, -np.inf], np.nan)
    return clean.where(pd.notnull(clean), None).to_dict(orient="records")


def _process_ticker(symbol: str, api_key: str, window_days: int = 90, PE_yr_range: int = 6) -> pd.DataFrame:
    """
    Core per-ticker computation. Returns stock_consolidate_df (window_days rows)
    with all computed metrics as columns, plus a DatetimeIndex.
    Raises on any fatal error.
    """
    # ── Stock split factor ────────────────────────────────────────────────
    url = f"https://www.alphavantage.co/query?function=SPLITS&symbol={symbol}&apikey={api_key}"
    data = requests.get(url).json()

    stock_split_record_df = pd.DataFrame({"split_factor": [1], "effective_date": [datetime.today()]})
    for key, value in data.items():
        if key == "data" and len(value) > 0:
            stock_split_record_df = pd.DataFrame(value)
            stock_split_record_df["split_factor"] = pd.to_numeric(
                stock_split_record_df["split_factor"], errors="coerce"
            )
            stock_split_record_df["effective_date"] = pd.to_datetime(stock_split_record_df["effective_date"])

    # ── Daily price ───────────────────────────────────────────────────────
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full"
    data = requests.get(url).json()

    Daily_stock_df = pd.DataFrame()
    for key, value in data.items():
        if key == "Time Series (Daily)":
            Daily_stock_df = pd.DataFrame(value).transpose()[["4. close"]]
            Daily_stock_df.rename(columns={"4. close": "stock_price"}, inplace=True)
            Daily_stock_df["stock_price"] = Daily_stock_df["stock_price"].astype(float).round(2)
            Daily_stock_df.index = pd.to_datetime(Daily_stock_df.index)

    if Daily_stock_df.empty:
        raise ValueError("No daily price data returned from Alpha Vantage")

    # Adjust for stock splits
    for date_i in Daily_stock_df.index.date:
        for date_j in stock_split_record_df["effective_date"].dt.date:
            if date_i == date_j:
                factor = stock_split_record_df.loc[
                    stock_split_record_df["effective_date"].dt.date == date_j, "split_factor"
                ].values[0]
                Daily_stock_df.loc[Daily_stock_df.index.date < date_j, "stock_price"] /= factor

    # MA200 & 20-day slope
    Daily_stock_df["MA200"] = (
        Daily_stock_df.sort_index(ascending=True)["stock_price"].rolling(window=200).mean()
    )

    def _slope(series):
        x = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        m = LinearRegression()
        m.fit(x, y)
        return m.coef_[0][0]

    Daily_stock_df["MA200_slope"] = (
        Daily_stock_df.sort_index(ascending=True)["MA200"]
        .rolling(window=20)
        .apply(_slope, raw=False)
    )

    # ── Monthly price (year-end for historical PE) ────────────────────────
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey={api_key}"
    data = requests.get(url).json()

    Monthly_stock_df = pd.DataFrame()
    for key, value in data.items():
        if key == "Monthly Time Series":
            Monthly_stock_df = pd.DataFrame(value).transpose()
            Monthly_stock_df.index = pd.to_datetime(Monthly_stock_df.index)

    f1 = Monthly_stock_df.index.year.isin(range(datetime.today().year - PE_yr_range, datetime.today().year))
    f2 = Monthly_stock_df.index.month == 12
    Monthly_stock_df = Monthly_stock_df[f1 & f2][["4. close"]].rename(columns={"4. close": "stock_price"})
    Monthly_stock_df["stock_price"] = Monthly_stock_df["stock_price"].astype(float).round(2)

    for year_i in Monthly_stock_df.index.year:
        for year_j in stock_split_record_df["effective_date"].dt.year:
            if year_i == year_j:
                factor = stock_split_record_df.loc[
                    stock_split_record_df["effective_date"].dt.year == year_j, "split_factor"
                ].values[0]
                Monthly_stock_df.loc[Monthly_stock_df.index.year < year_j, "stock_price"] /= factor

    # ── Earnings: Alpha Vantage ───────────────────────────────────────────
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}"
    data = requests.get(url).json()

    annualEPS_df = pd.DataFrame()
    qtrEPS_df = pd.DataFrame()

    for key, value in data.items():
        if key == "annualEarnings":
            annualEPS_df = pd.DataFrame(value)[["fiscalDateEnding", "reportedEPS"]]
            annualEPS_df["fiscalDateEnding"] = pd.to_datetime(annualEPS_df["fiscalDateEnding"]).dt.year
            annualEPS_df = annualEPS_df[
                annualEPS_df["fiscalDateEnding"].isin(
                    range(datetime.today().year - PE_yr_range, datetime.today().year)
                )
            ]
            annualEPS_df["reportedEPS"] = annualEPS_df["reportedEPS"].astype(str).apply(float)
            annualEPS_df = (
                annualEPS_df.sort_values("reportedEPS", ascending=False)
                .drop_duplicates("fiscalDateEnding")
                .sort_values("fiscalDateEnding", ascending=False)
                .reset_index(drop=True)
            )
            if Monthly_stock_df.shape[0] <= annualEPS_df.shape[0]:
                annualEPS_df = annualEPS_df[: Monthly_stock_df.shape[0]]
            annualEPS_df["PE"] = Monthly_stock_df["stock_price"].values / annualEPS_df["reportedEPS"].values
            annualEPS_df[f"PE_{PE_yr_range - 1}yr_avg"] = annualEPS_df["PE"].mean().round(2)
            annualEPS_df[f"PE_{PE_yr_range - 1}yr_std"] = np.std(annualEPS_df["PE"]).round(2)
            annualEPS_df[f"PE_{PE_yr_range - 1}yr_volatility_+"] = (
                annualEPS_df[f"PE_{PE_yr_range - 1}yr_avg"] + annualEPS_df[f"PE_{PE_yr_range - 1}yr_std"]
            ).round(2)
            annualEPS_df[f"PE_{PE_yr_range - 1}yr_volatility_-"] = (
                annualEPS_df[f"PE_{PE_yr_range - 1}yr_avg"] - annualEPS_df[f"PE_{PE_yr_range - 1}yr_std"]
            ).round(2)

        if key == "quarterlyEarnings":
            qtrEPS_df = pd.DataFrame(value)[["reportedDate", "reportedEPS"]]
            qtrEPS_df["reportedDate"] = pd.to_datetime(qtrEPS_df["reportedDate"])
            qtrEPS_df["reportedEPS"] = qtrEPS_df["reportedEPS"].astype(str).apply(
                lambda x: float(x) if x not in [None, "None", "nan", "NaN"] else 0.0
            )

    # ── EPS forecasts: Alpha Vantage EARNINGS_CALENDAR ────────────────────
    next_qtr_EPS = 0.0
    next_yr_EPS = 0.0

    cal_url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol={symbol}&horizon=12month&apikey={api_key}"
    with requests.Session() as s:
        decoded = s.get(cal_url).content.decode("utf-8")
        rows = list(csv_module.reader(decoded.splitlines(), delimiter=","))
        if len(rows) > 1:
            cal_df = pd.DataFrame(columns=rows[0], data=rows[1:])
            cal_df["estimate"] = pd.to_numeric(cal_df["estimate"], errors="coerce").fillna(0)
            if not cal_df.empty:
                next_qtr_EPS = float(cal_df["estimate"].iloc[0])
            next_yr_EPS = float(cal_df["estimate"].head(4).sum())

    # ── Growth / EPS trend via yfinance info ──────────────────────────────
    next_yr_days7ago_EPS = None
    next_yr_days30ago_EPS = None
    next_yr_days60ago_EPS = None
    next_yr_days90ago_EPS = None
    curr_yr_growthrate_symbol = None
    next_yr_growthrate_symbol = None
    curr_yr_growthrate_index = None
    next_yr_growthrate_index = None

    try:
        yf_info = yf.Ticker(symbol).info
        forward_eps = yf_info.get("forwardEps")
        if forward_eps and next_yr_EPS == 0:
            next_yr_EPS = float(forward_eps)
        earnings_growth = yf_info.get("earningsGrowth")
        if earnings_growth is not None:
            curr_yr_growthrate_symbol = round(float(earnings_growth) * 100, 2)
            next_yr_growthrate_symbol = round(float(earnings_growth) * 100, 2)
    except Exception:
        pass

    # ── US 10yr Treasury yield ────────────────────────────────────────────
    url = f"https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey={api_key}"
    data = requests.get(url).json()
    US_T_10yrs_YTM = None
    for key, value in data.items():
        if key == "data":
            t_df = pd.DataFrame(value)
            t_df["value"] = pd.to_numeric(t_df["value"], errors="coerce")
            US_T_10yrs_YTM = t_df["value"].iloc[0]

    # ── Company overview ──────────────────────────────────────────────────
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
    data = requests.get(url).json()
    float(data.get("MarketCapitalization", 0) or 0)  # kept for future use

    # ── Income statement ──────────────────────────────────────────────────
    url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={api_key}"
    data = requests.get(url).json()

    for key, value in data.items():
        if key == "annualReports":
            annual_income_df = pd.DataFrame(value).drop(["reportedCurrency"], axis=1)
            annual_income_df = annual_income_df.sort_values("fiscalDateEnding", ascending=True)
            for col in annual_income_df.columns[1:]:
                annual_income_df[col] = pd.to_numeric(annual_income_df[col], errors="coerce")
                annual_income_df[f"{col}_YoY"] = annual_income_df[col].pct_change() * 100
            annual_income_df["gross_margin_%"] = annual_income_df["grossProfit"] / annual_income_df["totalRevenue"] * 100
            annual_income_df["operating_margin_%"] = annual_income_df["operatingIncome"] / annual_income_df["totalRevenue"] * 100
            annual_income_df["net_margin_%"] = annual_income_df["netIncome"] / annual_income_df["totalRevenue"] * 100

        if key == "quarterlyReports":
            qtr_income_df = pd.DataFrame(value).drop(["reportedCurrency"], axis=1)
            qtr_income_df = qtr_income_df.sort_values("fiscalDateEnding", ascending=True)
            for col in qtr_income_df.columns[1:]:
                qtr_income_df[col] = pd.to_numeric(qtr_income_df[col], errors="coerce")
                qtr_income_df[f"{col}_QoQ"] = qtr_income_df[col].pct_change() * 100
            qtr_income_df["gross_margin_%"] = qtr_income_df["grossProfit"] / qtr_income_df["totalRevenue"] * 100
            qtr_income_df["operating_margin_%"] = qtr_income_df["operatingIncome"] / qtr_income_df["totalRevenue"] * 100
            qtr_income_df["net_margin_%"] = qtr_income_df["netIncome"] / qtr_income_df["totalRevenue"] * 100

    # ── Balance sheet ─────────────────────────────────────────────────────
    url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey={api_key}"
    data = requests.get(url).json()

    for key, value in data.items():
        if key == "annualReports":
            annual_balancesheet_df = pd.DataFrame(value).drop(["reportedCurrency"], axis=1)
            annual_balancesheet_df = annual_balancesheet_df.sort_values("fiscalDateEnding", ascending=True)
            for col in annual_balancesheet_df.columns[1:]:
                annual_balancesheet_df[col] = pd.to_numeric(annual_balancesheet_df[col], errors="coerce")

        if key == "quarterlyReports":
            qtr_balancesheet_df = pd.DataFrame(value).drop(["reportedCurrency"], axis=1)
            qtr_balancesheet_df = qtr_balancesheet_df.sort_values("fiscalDateEnding", ascending=True)
            for col in qtr_balancesheet_df.columns[1:]:
                qtr_balancesheet_df[col] = pd.to_numeric(qtr_balancesheet_df[col], errors="coerce")
            qtr_balancesheet_df["BVPS_latest"] = round(
                qtr_balancesheet_df.tail(1)["totalShareholderEquity"].sum()
                / qtr_balancesheet_df.tail(1)["commonStockSharesOutstanding"],
                2,
            )

    # ── Cash flow ─────────────────────────────────────────────────────────
    url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={api_key}"
    data = requests.get(url).json()

    for key, value in data.items():
        if key == "quarterlyReports":
            qtr_cashflow_df = pd.DataFrame(value).drop(["reportedCurrency"], axis=1)
            qtr_cashflow_df = qtr_cashflow_df.sort_values("fiscalDateEnding", ascending=True)
            for col in qtr_cashflow_df.columns[1:]:
                qtr_cashflow_df[col] = pd.to_numeric(qtr_cashflow_df[col], errors="coerce")

    # ── Build daily consolidated df ───────────────────────────────────────
    df = Daily_stock_df.head(window_days).copy()

    EPS_nextQtr_projected = 0.0
    for i in df.index:
        past = qtrEPS_df[qtrEPS_df["reportedDate"] < i]
        df.loc[i, "EPS_TTM"] = past.head(4)["reportedEPS"].sum()
        df.loc[i, "EPS_currentQtr"] = past.head(1)["reportedEPS"].sum()
        if i == df.index.max():
            EPS_nextQtr_projected = next_qtr_EPS + past.head(3)["reportedEPS"].sum()

    df["Ticker"] = symbol
    df["EPS_nextQtr"] = next_qtr_EPS
    df["EPS_nextQtr_TTM"] = EPS_nextQtr_projected
    df["EPS_nextYr"] = round(next_yr_EPS, 2)

    # PE TTM metrics
    df["PE_TTM"] = (df["stock_price"] / df["EPS_TTM"]).round(2)
    df["PE_TTM_avg"] = df["PE_TTM"].mean().round(2)
    df["PE_TTM_std"] = np.std(df["PE_TTM"]).round(2)
    df["PE_TTM_volatility_+"] = (df["PE_TTM_avg"] + df["PE_TTM_std"]).round(2)
    df["PE_TTM_volatility_-"] = (df["PE_TTM_avg"] - df["PE_TTM_std"]).round(2)

    # Historical PE (n-year)
    df[f"PE_{PE_yr_range - 1}yr_avg"] = annualEPS_df[f"PE_{PE_yr_range - 1}yr_avg"].values[0]
    df[f"PE_{PE_yr_range - 1}yr_volatility_+"] = annualEPS_df[f"PE_{PE_yr_range - 1}yr_volatility_+"].values[0]
    df[f"PE_{PE_yr_range - 1}yr_volatility_-"] = annualEPS_df[f"PE_{PE_yr_range - 1}yr_volatility_-"].values[0]

    # Relative valuation ranges
    df["relative_valuation_TTM_+"] = (df["PE_TTM_volatility_+"] * df["EPS_TTM"]).round(2)
    df["relative_valuation_TTM_-"] = (df["PE_TTM_volatility_-"] * df["EPS_TTM"]).round(2)
    df["relative_valuation_TTM_median"] = np.median(
        [df["relative_valuation_TTM_+"].iloc[0], df["relative_valuation_TTM_-"].iloc[0]]
    ).round(2)

    df["relative_valuation_nextQuater_projected_+"] = (df["PE_TTM_volatility_+"] * df["EPS_nextQtr_TTM"]).round(2)
    df["relative_valuation_nextQuater_projected_-"] = (df["PE_TTM_volatility_-"] * df["EPS_nextQtr_TTM"]).round(2)
    df["relative_valuation_nextQuater_projected_median"] = np.median(
        [df["relative_valuation_nextQuater_projected_+"].iloc[0], df["relative_valuation_nextQuater_projected_-"].iloc[0]]
    ).round(2)

    df["relative_valuation_nextYear_projected_+"] = (df["PE_TTM_volatility_+"] * next_yr_EPS).round(2)
    df["relative_valuation_nextYear_projected_-"] = (df["PE_TTM_volatility_-"] * next_yr_EPS).round(2)
    df["relative_valuation_nextYear_projected_median"] = np.median(
        [df["relative_valuation_nextYear_projected_+"].iloc[0], df["relative_valuation_nextYear_projected_-"].iloc[0]]
    ).round(2)

    # Price window stats
    df[f"{window_days}_price_min"] = df["stock_price"].min().round(2)
    df[f"{window_days}_price_max"] = df["stock_price"].max().round(2)
    df[f"{window_days}_price_avg"] = df["stock_price"].mean().round(2)
    df[f"{window_days}_price_std"] = np.std(df["stock_price"]).round(2)

    # Growth & yield
    df["EPS_nextYr_growthRate"] = (((next_yr_EPS - df["EPS_TTM"]) / df["EPS_TTM"]) * 100).round(2)
    df["EPS_nextQtr_growthRate"] = (((df["EPS_nextQtr_TTM"] - df["EPS_TTM"]) / df["EPS_TTM"]) * 100).round(2)
    df["EarningYield_TTM"] = ((df["EPS_TTM"] / df["stock_price"]) * 100).round(2)
    df["ERP_TTM"] = (df["EarningYield_TTM"] - US_T_10yrs_YTM).round(2) if US_T_10yrs_YTM is not None else None

    # Balance sheet & cash flow derived
    df["BVPS_latest"] = qtr_balancesheet_df["BVPS_latest"].values[-1]
    df["MA200_slope"] = Daily_stock_df["MA200_slope"]
    df["FCF_per_share_TTM"] = round(
        (qtr_cashflow_df["operatingCashflow"] - qtr_cashflow_df["capitalExpenditures"]).tail(4).sum()
        / qtr_balancesheet_df["commonStockSharesOutstanding"].values[-1],
        2,
    )
    df["PFCF_TTM"] = round(df["stock_price"].iloc[0] / df["FCF_per_share_TTM"], 2)
    df["FCF_yield_TTM"] = round((df["FCF_per_share_TTM"] / df["stock_price"].iloc[0]) * 100, 2)

    # EPS trend (unavailable in new yfinance — stored as None)
    df["next_yr_days7ago_EPS"] = _safe_round(next_yr_days7ago_EPS)
    df["next_yr_days30ago_EPS"] = _safe_round(next_yr_days30ago_EPS)
    df["next_yr_days60ago_EPS"] = _safe_round(next_yr_days60ago_EPS)
    df["next_yr_days90ago_EPS"] = _safe_round(next_yr_days90ago_EPS)

    # Growth rates
    df["curr_yr_growthrate_symbol"] = _safe_round(curr_yr_growthrate_symbol)
    df["next_yr_growthrate_symbol"] = _safe_round(next_yr_growthrate_symbol)
    df["curr_yr_growthrate_index"] = _safe_round(curr_yr_growthrate_index)
    df["next_yr_growthrate_index"] = _safe_round(next_yr_growthrate_index)

    # Valuation assessment
    df["price_valuation_assessment"] = None
    for condition, category in [
        (df["stock_price"] < df["relative_valuation_TTM_-"], "undervalued"),
        (df["stock_price"] > df["relative_valuation_TTM_+"], "overvalued"),
        (
            (df["stock_price"] >= df["relative_valuation_TTM_-"])
            & (df["stock_price"] <= df["relative_valuation_TTM_+"]),
            "fair",
        ),
    ]:
        df.loc[condition, "price_valuation_assessment"] = category

    return df


# ── Public API ────────────────────────────────────────────────────────────────

def process_ticker_to_row(symbol: str, api_key: str, window_days: int = 90, PE_yr_range: int = 6) -> dict:
    """
    Process a single ticker and return its screen-row dict.
    Raises on error — callers should catch.
    """
    import math

    df = _process_ticker(symbol, api_key, window_days, PE_yr_range)
    n = PE_yr_range - 1
    dyn_cols = (
        SCREEN_COLS[:6]
        + [f"PE_{n}yr_avg", f"PE_{n}yr_volatility_+", f"PE_{n}yr_volatility_-"]
        + SCREEN_COLS[6:]
    )
    existing = [c for c in dyn_cols if c in df.columns]
    row = df[existing].iloc[0].to_dict()
    return {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v) for k, v in row.items()}


def analyze_stocks(ticker_symbols: list, api_key: str, window_days: int = 90, PE_yr_range: int = 6) -> dict:
    """
    Screen a list of tickers. Returns one summary row per ticker.
    Response: { screen: [...], errors: [...] }
    """
    screen_rows = []
    errors = []

    for j, symbol in enumerate(ticker_symbols):
        try:
            df = _process_ticker(symbol, api_key, window_days, PE_yr_range)

            # Build dynamic screen cols (include the n-yr PE cols)
            dyn_cols = (
                SCREEN_COLS[:6]
                + [f"PE_{PE_yr_range - 1}yr_avg", f"PE_{PE_yr_range - 1}yr_volatility_+", f"PE_{PE_yr_range - 1}yr_volatility_-"]
                + SCREEN_COLS[6:]
            )
            existing = [c for c in dyn_cols if c in df.columns]
            row = df[existing].iloc[0].to_dict()
            screen_rows.append(row)
        except Exception as e:
            errors.append({"ticker": symbol, "message": str(e)})

    # Add industry average PE
    if screen_rows:
        pe_vals = [r.get("PE_TTM") for r in screen_rows if r.get("PE_TTM") is not None]
        industry_avg = round(float(np.mean([v for v in pe_vals if v == v])), 2) if pe_vals else None
        for r in screen_rows:
            r["Industry_PE_TTM_avg"] = industry_avg

    # Sanitize NaN/inf
    import math
    def _sanitize(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    clean_rows = [{k: _sanitize(v) for k, v in row.items()} for row in screen_rows]
    return {"screen": clean_rows, "errors": errors}


def analyze_single_stock_detail(symbol: str, api_key: str, window_days: int = 90, PE_yr_range: int = 6) -> dict:
    """
    Return the full window_days daily history for one ticker (mirrors ticker_dict_pd[symbol]).
    Response: { ticker, columns, rows, errors }
      - rows: list of { date, ...metrics }
    """
    try:
        df = _process_ticker(symbol, api_key, window_days, PE_yr_range)

        # Build detail cols including the dynamic PE n-yr columns
        dyn_detail_cols = (
            DETAIL_COLS[:10]
            + [f"PE_{PE_yr_range - 1}yr_avg", f"PE_{PE_yr_range - 1}yr_volatility_+", f"PE_{PE_yr_range - 1}yr_volatility_-"]
            + DETAIL_COLS[10:]
        )
        existing = [c for c in dyn_detail_cols if c in df.columns]
        out_df = df[existing].copy()

        # Include the date as a column
        out_df.index = out_df.index.strftime("%Y-%m-%d")
        out_df.index.name = "date"
        out_df = out_df.reset_index()

        records = _clean_df_for_json(out_df)
        return {"ticker": symbol, "columns": list(out_df.columns), "rows": records, "error": None}

    except Exception as e:
        return {"ticker": symbol, "columns": [], "rows": [], "error": str(e)}
