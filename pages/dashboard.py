"""
Dashboard page — real-time overview, company profile, key metrics.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_fetcher import get_stock_info, get_current_price, get_historical_data
from utils.charts import candlestick_chart
from utils.technical_indicators import calculate_sma, detect_trend, get_trading_signals


def render(ticker: str):
    st.markdown('<h1 class="main-header">📈 Stock Dashboard</h1>', unsafe_allow_html=True)

    if not ticker:
        st.warning("Please enter a ticker symbol in the sidebar.")
        return

    with st.spinner(f"Loading data for {ticker}..."):
        info = get_stock_info(ticker)
        price_data = get_current_price(ticker)
        df = get_historical_data(ticker, period="3mo")

    if not info and not price_data:
        st.error(f"Could not find data for '{ticker}'. Please check the ticker symbol.")
        return

    company_name = info.get("longName") or info.get("shortName") or ticker
    st.markdown(f"## {company_name} ({ticker})")

    # ── Real-time Price Strip ──────────────────────────────────────────────
    if price_data:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        price = price_data.get("price", 0)
        change = price_data.get("change", 0)
        pct = price_data.get("pct_change", 0)
        change_color = "positive" if change >= 0 else "negative"
        arrow = "▲" if change >= 0 else "▼"

        col1.metric("Current Price", f"${price:,.2f}",
                    f"{arrow} {abs(change):.2f} ({pct:+.2f}%)")
        col2.metric("Open", f"${price_data.get('open', 0):,.2f}")
        col3.metric("Day High", f"${price_data.get('high', 0):,.2f}")
        col4.metric("Day Low", f"${price_data.get('low', 0):,.2f}")
        vol = price_data.get("volume", 0)
        col5.metric("Volume", f"{vol:,}")
        mkt_cap = info.get("marketCap", 0)
        if mkt_cap:
            if mkt_cap >= 1e12:
                mkt_cap_str = f"${mkt_cap/1e12:.2f}T"
            elif mkt_cap >= 1e9:
                mkt_cap_str = f"${mkt_cap/1e9:.2f}B"
            else:
                mkt_cap_str = f"${mkt_cap/1e6:.2f}M"
        else:
            mkt_cap_str = "N/A"
        col6.metric("Market Cap", mkt_cap_str)
    else:
        st.warning("Real-time price data unavailable.")

    st.markdown("---")

    # ── Price Chart (3 months) ─────────────────────────────────────────────
    if not df.empty:
        col_chart, col_info = st.columns([2, 1])
        with col_chart:
            indicators = {
                "sma20": calculate_sma(df, 20),
                "sma50": calculate_sma(df, 50),
            }
            fig = candlestick_chart(df, ticker, indicators)
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            # Trading signals
            signals = get_trading_signals(df)
            signal_color = {
                "STRONG BUY": "#00c853",
                "BUY": "#8bc34a",
                "HOLD": "#ffd600",
                "SELL": "#ff9800",
                "STRONG SELL": "#ff1744",
            }.get(signals["signal"], "#aaa")

            st.markdown(f"""
            <div style="background:#1e1e2e;border-radius:12px;padding:1rem;border:1px solid #333;text-align:center;">
                <div style="font-size:0.9rem;color:#aaa;">Trading Signal</div>
                <div style="font-size:2rem;font-weight:700;color:{signal_color};">{signals['signal']}</div>
                <div style="font-size:0.8rem;color:#aaa;">Score: {signals['score']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Signal Breakdown**")
            for reason in signals["reasons"]:
                icon = "🟢" if "Bullish" in reason else ("🔴" if "Bearish" in reason else "🟡")
                st.caption(f"{icon} {reason}")

            trend = detect_trend(df)
            st.markdown(f"**Overall Trend:** {trend}")
    
    st.markdown("---")

    # ── Company Profile ────────────────────────────────────────────────────
    st.markdown("### 🏢 Company Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Company Details**")
        details = {
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Exchange": info.get("exchange", "N/A"),
            "Country": info.get("country", "N/A"),
            "Employees": f"{info.get('fullTimeEmployees', 0):,}" if info.get("fullTimeEmployees") else "N/A",
            "Website": info.get("website", "N/A"),
        }
        for k, v in details.items():
            st.markdown(f"**{k}:** {v}")

    with col2:
        st.markdown("**Key Statistics**")
        stats = {
            "52W High": f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get("fiftyTwoWeekHigh") else "N/A",
            "52W Low": f"${info.get('fiftyTwoWeekLow', 0):.2f}" if info.get("fiftyTwoWeekLow") else "N/A",
            "Beta": f"{info.get('beta', 'N/A')}",
            "P/E Ratio": f"{info.get('trailingPE', 'N/A')}",
            "EPS (TTM)": f"${info.get('trailingEps', 'N/A')}",
            "Dividend Yield": f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get("dividendYield") else "N/A",
        }
        for k, v in stats.items():
            st.markdown(f"**{k}:** {v}")

    with col3:
        st.markdown("**Analyst Recommendations**")
        target_price = info.get("targetMeanPrice")
        current = price_data.get("price", 0) if price_data else 0
        if target_price and current:
            upside = ((target_price - current) / current) * 100
            upside_color = "#00c853" if upside > 0 else "#ff1744"
            st.markdown(f"**Target Price:** ${target_price:.2f}")
            st.markdown(f"**Upside/Downside:** <span style='color:{upside_color}'>{upside:+.1f}%</span>",
                        unsafe_allow_html=True)
        rec = info.get("recommendationKey", "N/A").upper()
        st.markdown(f"**Consensus:** {rec}")
        st.markdown(f"**Buy:** {info.get('numberOfAnalystOpinions', 'N/A')} analysts")

    # ── Business Summary ───────────────────────────────────────────────────
    summary = info.get("longBusinessSummary", "")
    if summary:
        with st.expander("📋 Business Summary"):
            st.write(summary)
