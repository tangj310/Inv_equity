#!/usr/bin/env python3
"""
export_portfolio.py
-------------------
Reads the last-executed HTML output of inv_summary_01.ipynb,
extracts the inv_yrly_merged_consolidate table, and writes
portfolio_returns.json in this same directory for the FastAPI home page.

No extra dependencies — uses only Python stdlib + json (already in project).

Usage (from the project root):
    python 01_stock_search/export_portfolio.py

Or from inside 01_stock_search/:
    python export_portfolio.py

Prerequisites:
    Run (and save) inv_summary_01.ipynb first so its cell outputs are populated.
"""

import json
import math
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent.resolve()
NOTEBOOK  = _HERE.parent / "inv_summary_01.ipynb"
OUTPUT    = _HERE / "portfolio_returns.json"


# ── Minimal HTML table parser (no lxml / beautifulsoup4 needed) ────────────
class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._headers   = []
        self._rows      = []
        self._cur_row   = []
        self._cur_cell  = None
        self._in_thead  = False
        self._in_tbody  = False

    def handle_starttag(self, tag, attrs):
        if tag == "thead":        self._in_thead = True
        if tag == "tbody":        self._in_tbody = True
        if tag in ("th", "td"):   self._cur_cell = ""

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self._cur_cell is not None:
            val = self._cur_cell.strip()
            if self._in_thead:
                self._headers.append(val)
            elif self._in_tbody:
                self._cur_row.append(val)
            self._cur_cell = None
        if tag == "tr":
            if self._in_tbody and self._cur_row:
                self._rows.append(list(self._cur_row))
                self._cur_row.clear()
        if tag == "thead": self._in_thead = False
        if tag == "tbody": self._in_tbody = False

    def handle_data(self, data):
        if self._cur_cell is not None:
            self._cur_cell += data

    def to_records(self):
        # headers[0] is the pandas row-index column — skip it
        cols = self._headers[1:]
        records = []
        for row in self._rows:
            values = row[1:]            # skip index value
            rec = {}
            for col, val in zip(cols, values):
                try:
                    rec[col] = float(val)
                except (ValueError, TypeError):
                    rec[col] = val if val != "" else None
            records.append(rec)
        return records


# ── Helpers ────────────────────────────────────────────────────────────────
def _safe(v):
    """Convert NaN / inf to None for JSON serialisation."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _extract_table(nb_path: Path):
    """
    Scan notebook cells for the one whose source is exactly
    'inv_yrly_merged_consolidate', then parse its HTML output.
    Returns list-of-dicts or None if not found.
    """
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        src = "".join(cell.get("source", [])).strip()
        if src != "inv_yrly_merged_consolidate":
            continue

        for out in cell.get("outputs", []):
            html = out.get("data", {}).get("text/html", "")
            if isinstance(html, list):
                html = "".join(html)
            if "Activity_Yr" not in html:
                continue

            parser = _TableParser()
            parser.feed(html)
            records = parser.to_records()
            if records:
                return records

    return None


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    if not NOTEBOOK.exists():
        raise FileNotFoundError(
            f"Notebook not found: {NOTEBOOK}\n"
            "Make sure you are running from the project root or 01_stock_search/."
        )

    print(f"Reading: {NOTEBOOK}")
    records = _extract_table(NOTEBOOK)

    if not records:
        print(
            "\nERROR: could not find 'inv_yrly_merged_consolidate' output in the notebook.\n"
            "Steps to fix:\n"
            "  1. Open inv_summary_01.ipynb in Jupyter.\n"
            "  2. Run all cells (Kernel → Restart & Run All).\n"
            "  3. Save the notebook (Ctrl+S).\n"
            "  4. Re-run this script.\n"
        )
        return

    # Scalar metrics — same value repeated every row; pull from first row
    cagr    = _safe(records[0].get("CAGR_%"))
    sp500c  = _safe(records[0].get("sp500_CAGR_%"))
    mwr     = _safe(records[0].get("MWR_IRR_%"))

    # Per-year rows (only the columns the home page table needs)
    rows = []
    for r in records:
        rows.append({
            "Activity_Yr":           int(r["Activity_Yr"]),
            "Ratio_Yearly_Return_%": _safe(r.get("Ratio_Yearly_Return_%")),
            "sp500_without_div_%":   _safe(r.get("sp500_without_div_%")),
        })

    payload = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "CAGR_%":       cagr,
        "sp500_CAGR_%": sp500c,
        "MWR_IRR_%":    mwr,
        "rows":         rows,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Written:  {OUTPUT}")
    print(f"  Rows:        {len(rows)}  ({rows[0]['Activity_Yr']} – {rows[-1]['Activity_Yr']})")
    print(f"  CAGR:        {cagr}%")
    print(f"  S&P 500 CAGR:{sp500c}%")
    print(f"  MWR / IRR:   {mwr}%")
    print(f"  Updated:     {payload['last_updated']}")


if __name__ == "__main__":
    main()
