"""
Simple in-memory cache with TTL for API results.
Avoids re-fetching the same ticker from Alpha Vantage / yfinance
within CACHE_TTL_HOURS (default 4 hours).
"""

import time
import os

CACHE_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", "4"))
_CACHE_TTL = CACHE_TTL_HOURS * 3600  # seconds

_store: dict[str, tuple] = {}  # key -> (value, timestamp)


def cache_get(key: str):
    """Return cached value if it exists and hasn't expired, else None."""
    if key in _store:
        value, ts = _store[key]
        if time.time() - ts < _CACHE_TTL:
            return value
        del _store[key]
    return None


def cache_set(key: str, value):
    """Store a value with the current timestamp."""
    _store[key] = (value, time.time())


def cache_clear():
    """Clear all cached entries."""
    _store.clear()


def cache_stats() -> dict:
    """Return basic cache info."""
    now = time.time()
    active = sum(1 for _, (__, ts) in _store.items() if now - ts < _CACHE_TTL)
    return {"total_entries": len(_store), "active_entries": active, "ttl_hours": CACHE_TTL_HOURS}
