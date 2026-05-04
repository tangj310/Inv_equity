import asyncio
import json
import math
import os

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from stock_analyzer import analyze_stocks, analyze_single_stock_detail, process_ticker_to_row
from stock_comparison import process_ticker_return
from cache import cache_clear, cache_stats

load_dotenv()

app = FastAPI(title="PE TTM Stock Valuation")
templates = Jinja2Templates(directory="templates")

API_KEY = os.getenv("alpha_vantage_api_key", "")


class AnalyzeRequest(BaseModel):
    tickers: list[str]
    window_days: int = 90
    PE_yr_range: int = 6


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/stock-screening", response_class=HTMLResponse)
async def stock_screening(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze(body: AnalyzeRequest):
    if not API_KEY:
        return {"error": "alpha_vantage_api_key not set in environment", "screen": [], "errors": []}

    tickers = [t.strip().upper() for t in body.tickers if t.strip()]
    if not tickers:
        return {"screen": [], "errors": []}

    result = analyze_stocks(
        ticker_symbols=tickers,
        api_key=API_KEY,
        window_days=body.window_days,
        PE_yr_range=body.PE_yr_range,
    )
    return result


@app.post("/analyze-stream")
async def analyze_stream(body: AnalyzeRequest):
    """SSE endpoint: streams one progress/result/error event per ticker, then a final 'done' event."""
    if not API_KEY:
        async def _err():
            yield f"data: {json.dumps({'type': 'error_fatal', 'message': 'alpha_vantage_api_key not set'})}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    tickers = [t.strip().upper() for t in body.tickers if t.strip()]

    async def event_generator():
        screen_rows = []
        errors = []
        total = len(tickers)
        loop = asyncio.get_event_loop()

        for j, symbol in enumerate(tickers):
            # ── "fetching" progress event ──────────────────────────────────
            yield f"data: {json.dumps({'type': 'progress', 'ticker': symbol, 'index': j + 1, 'total': total})}\n\n"

            try:
                row = await loop.run_in_executor(
                    None, process_ticker_to_row, symbol, API_KEY, body.window_days, body.PE_yr_range
                )
                screen_rows.append(row)
                yield f"data: {json.dumps({'type': 'result', 'ticker': symbol, 'index': j + 1, 'total': total, 'row': row})}\n\n"
            except Exception as e:
                err = {"ticker": symbol, "message": str(e)}
                errors.append(err)
                yield f"data: {json.dumps({'type': 'error', 'ticker': symbol, 'index': j + 1, 'total': total, 'message': str(e)})}\n\n"

        # ── Finalize: add industry PE avg ──────────────────────────────────
        if screen_rows:
            pe_vals = [r.get("PE_TTM") for r in screen_rows if r.get("PE_TTM") is not None]
            valid = [v for v in pe_vals if not (isinstance(v, float) and math.isnan(v))]
            industry_avg = round(float(np.mean(valid)), 2) if valid else None
            for r in screen_rows:
                r["Industry_PE_TTM_avg"] = industry_avg

        yield f"data: {json.dumps({'type': 'done', 'screen': screen_rows, 'errors': errors})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class SingleStockRequest(BaseModel):
    ticker: str
    window_days: int = 90
    PE_yr_range: int = 6


@app.get("/single-stock", response_class=HTMLResponse)
async def single_stock_page(request: Request):
    return templates.TemplateResponse("single_stock.html", {"request": request})


@app.post("/single-stock-data")
async def single_stock_data(body: SingleStockRequest):
    if not API_KEY:
        return {"ticker": body.ticker, "columns": [], "rows": [], "error": "alpha_vantage_api_key not set"}
    ticker = body.ticker.strip().upper()
    return analyze_single_stock_detail(ticker, API_KEY, body.window_days, body.PE_yr_range)


class ReturnComparisonRequest(BaseModel):
    tickers: list[str]
    years: float = 5.0


@app.get("/stock-return-comparison", response_class=HTMLResponse)
async def stock_return_comparison_page(request: Request):
    return templates.TemplateResponse("stock_comparison.html", {"request": request})


@app.post("/stock-return-stream")
async def stock_return_stream(body: ReturnComparisonRequest):
    """SSE: streams one result per ticker, then 'done'."""
    if not API_KEY:
        async def _err():
            yield f"data: {json.dumps({'type': 'error_fatal', 'message': 'alpha_vantage_api_key not set'})}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    tickers = [t.strip().upper() for t in body.tickers if t.strip()]

    async def event_generator():
        total = len(tickers)
        loop  = asyncio.get_event_loop()
        results = []

        for j, symbol in enumerate(tickers):
            yield f"data: {json.dumps({'type': 'progress', 'ticker': symbol, 'index': j + 1, 'total': total})}\n\n"
            try:
                result = await loop.run_in_executor(
                    None, process_ticker_return, symbol, body.years, API_KEY
                )
                results.append(result)
                yield f"data: {json.dumps({'type': 'result', 'ticker': symbol, 'index': j + 1, 'total': total, 'data': result})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'ticker': symbol, 'index': j + 1, 'total': total, 'message': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/portfolio-returns")
async def portfolio_returns():
    """Serve pre-computed portfolio return data exported from inv_summary_01.ipynb."""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_returns.json")
    if not os.path.exists(data_path):
        return {"rows": [], "CAGR_%": None, "sp500_CAGR_%": None, "MWR_IRR_%": None, "last_updated": None}
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/cache-stats")
async def get_cache_stats():
    return cache_stats()


@app.post("/api/cache-clear")
async def clear_cache():
    cache_clear()
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
