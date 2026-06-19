"""
Data fetching utilities — direct Yahoo Finance API
(bypasses yfinance auth issues with Python 3.8 + yfinance 0.2.x)
"""

import pandas as pd
import numpy as np
import streamlit as st
import requests
import time
import random
from datetime import datetime

# ── shared session ────────────────────────────────────────────────────────────

_SESSION = None

def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
    return _SESSION


def _get(url: str, params: dict = None, retries: int = 3) -> dict:
    """GET with retry/back-off on 429."""
    session = _get_session()
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=15)
            if r.status_code == 429:
                wait = 2 ** attempt + random.uniform(0, 1)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1)
    return {}


# ── chart API (OHLCV) ─────────────────────────────────────────────────────────

_PERIOD_MAP = {
    "1d": "1d", "5d": "5d", "1mo": "1mo", "3mo": "3mo",
    "6mo": "6mo", "1y": "1y", "2y": "2y", "5y": "5y",
    "10y": "10y", "ytd": "ytd", "max": "max",
}


def _fetch_chart(ticker: str, period: str = "1y", interval: str = "1d",
                 start: str = None, end: str = None) -> tuple:
    """
    Returns (df, meta) where df is an OHLCV DataFrame and meta is
    the chart metadata dict (symbol, regularMarketPrice, 52-week range, etc.)
    """
    params = {"interval": interval, "events": "div,splits"}
    if start and end:
        params["period1"] = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        params["period2"] = int(datetime.strptime(end, "%Y-%m-%d").timestamp())
    else:
        params["range"] = _PERIOD_MAP.get(period, "1y")

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker.upper()}"
    data = _get(url, params=params)

    result = data.get("chart", {}).get("result")
    if not result:
        return pd.DataFrame(), {}

    r = result[0]
    meta = r.get("meta", {})
    timestamps = r.get("timestamp", [])
    if not timestamps:
        return pd.DataFrame(), meta

    quotes = r["indicators"]["quote"][0]
    adj_list = r["indicators"].get("adjclose", [{}])
    adj_close = adj_list[0].get("adjclose", quotes["close"]) if adj_list else quotes["close"]

    df = pd.DataFrame({
        "Open":      quotes.get("open", []),
        "High":      quotes.get("high", []),
        "Low":       quotes.get("low", []),
        "Close":     quotes.get("close", []),
        "Adj Close": adj_close,
        "Volume":    quotes.get("volume", []),
    }, index=pd.to_datetime(timestamps, unit="s"))

    df.index.name = "Date"
    df.dropna(subset=["Close"], inplace=True)
    return df, meta


def _fetch_search_info(ticker: str) -> dict:
    """Get sector/industry/name from the search endpoint (no auth needed)."""
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": ticker, "quotesCount": 1, "newsCount": 0}
        data = _get(url, params=params)
        quotes = data.get("quotes", [])
        if quotes:
            q = quotes[0]
            if q.get("symbol", "").upper() == ticker.upper():
                return q
    except Exception:
        pass
    return {}


# ── public API ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_stock_info(ticker: str) -> dict:
    """
    Fetch company info and current price data.
    Combines chart meta + search endpoint to populate a yfinance-like info dict.
    """
    ticker = ticker.upper().strip()
    try:
        df, meta = _fetch_chart(ticker, period="5d", interval="1d")
        if meta.get("instrumentType") not in ("EQUITY", "ETF", "MUTUALFUND", None) and not meta:
            st.error(f"Could not find data for '{ticker}'. Please check the ticker symbol.")
            return {}
        if df.empty and not meta:
            st.error(f"Could not find data for '{ticker}'. Please check the ticker symbol.")
            return {}

        search = _fetch_search_info(ticker)

        # Build a flat info dict resembling yfinance's stock.info
        info = {
            # Identity
            "symbol":            meta.get("symbol", ticker),
            "shortName":         meta.get("shortName") or search.get("shortname", ticker),
            "longName":          meta.get("longName") or search.get("longname", ticker),
            "sector":            search.get("sector", "N/A"),
            "industry":          search.get("industry", "N/A"),
            "exchange":          meta.get("fullExchangeName", meta.get("exchangeName", "N/A")),
            "currency":          meta.get("currency", "USD"),
            "quoteType":         meta.get("instrumentType", search.get("quoteType", "EQUITY")),
            # Price
            "currentPrice":      meta.get("regularMarketPrice"),
            "regularMarketPrice":meta.get("regularMarketPrice"),
            "previousClose":     meta.get("chartPreviousClose"),
            "open":              meta.get("regularMarketDayOpen"),
            "dayHigh":           meta.get("regularMarketDayHigh"),
            "dayLow":            meta.get("regularMarketDayLow"),
            "volume":            meta.get("regularMarketVolume"),
            "fiftyTwoWeekHigh":  meta.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow":   meta.get("fiftyTwoWeekLow"),
            # Calculated from price history
            "regularMarketChangePercent": None,
        }

        # Calculate day change %
        if info["currentPrice"] and info["previousClose"]:
            chg = info["currentPrice"] - info["previousClose"]
            info["regularMarketChange"] = round(chg, 2)
            info["regularMarketChangePercent"] = round((chg / info["previousClose"]) * 100, 2)

        # Fill in latest OHLCV from df if meta was sparse
        if not df.empty:
            last = df.iloc[-1]
            if not info["currentPrice"]:
                info["currentPrice"] = float(last["Close"])
                info["regularMarketPrice"] = float(last["Close"])
            if not info["volume"]:
                info["volume"] = int(last["Volume"])

        return info

    except Exception as e:
        st.error(f"Error fetching info for {ticker}: {e}")
        return {}


@st.cache_data(ttl=300)
def get_current_price(ticker: str) -> dict:
    """Get the latest price data."""
    ticker = ticker.upper().strip()
    try:
        df, meta = _fetch_chart(ticker, period="5d", interval="1d")
        if df.empty:
            return {}
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        change = latest["Close"] - prev["Close"]
        pct_change = (change / prev["Close"]) * 100
        return {
            "price":      round(float(latest["Close"]), 2),
            "change":     round(float(change), 2),
            "pct_change": round(float(pct_change), 2),
            "volume":     int(latest["Volume"]) if latest["Volume"] else 0,
            "high":       round(float(latest["High"]), 2),
            "low":        round(float(latest["Low"]), 2),
            "open":       round(float(latest["Open"]), 2),
        }
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def get_historical_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical OHLCV data."""
    ticker = ticker.upper().strip()
    try:
        df, _ = _fetch_chart(ticker, period=period, interval=interval)
        if df.empty:
            st.error(f"No historical data found for '{ticker}'.")
        return df
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_custom_range_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch data for a custom date range."""
    ticker = ticker.upper().strip()
    try:
        df, _ = _fetch_chart(ticker, interval=interval, start=start, end=end)
        if df.empty:
            st.error(f"No data found for '{ticker}' in the selected date range.")
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_financials(ticker: str) -> dict:
    """Fetch financial statements via yfinance (fallback)."""
    empty = {"income_stmt": pd.DataFrame(), "balance_sheet": pd.DataFrame(),
             "cash_flow": pd.DataFrame(), "earnings": pd.DataFrame()}
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker.upper())
        result = {}
        for attr, key in [
            ("financials",    "income_stmt"),
            ("balance_sheet", "balance_sheet"),
            ("cashflow",      "cash_flow"),
        ]:
            try:
                val = getattr(stock, attr)
                result[key] = val if val is not None else pd.DataFrame()
            except Exception:
                result[key] = pd.DataFrame()
        result["earnings"] = pd.DataFrame()
        return result
    except Exception:
        return empty


@st.cache_data(ttl=3600)
def get_multiple_stocks(tickers: list, period: str = "1y") -> dict:
    """Fetch closing prices for multiple tickers."""
    data = {}
    for t in tickers:
        try:
            df, _ = _fetch_chart(t.upper(), period=period)
            if not df.empty:
                data[t] = df["Close"]
            time.sleep(0.3)
        except Exception:
            pass
    return data


def validate_ticker(ticker: str) -> bool:
    """Check if a ticker is valid by attempting to fetch 5 days of data."""
    try:
        df, meta = _fetch_chart(ticker.upper(), period="5d")
        return not df.empty or bool(meta.get("symbol"))
    except Exception:
        return False
