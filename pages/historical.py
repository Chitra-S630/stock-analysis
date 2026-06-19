"""
Historical Data Analysis page — OHLCV charts, performance, custom date ranges.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
from utils.data_fetcher import get_historical_data, get_custom_range_data
from utils.charts import candlestick_chart, CHART_THEME


def render(ticker: str):
    st.markdown('<h1 class="main-header">📊 Historical Analysis</h1>', unsafe_allow_html=True)

    if not ticker:
        st.warning("Enter a ticker symbol in the sidebar.")
        return

    # ── Controls ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        period_option = st.selectbox(
            "Quick Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "Custom"],
        )
    with col2:
        interval = st.selectbox(
            "Interval",
            ["1d", "1wk", "1mo"],
            format_func=lambda x: {"1d": "Daily", "1wk": "Weekly", "1mo": "Monthly"}[x],
        )
    with col3:
        chart_type = st.selectbox("Chart Type", ["Candlestick", "OHLC", "Line", "Area"])

    custom_start = custom_end = None
    if period_option == "Custom":
        col_s, col_e = st.columns(2)
        with col_s:
            custom_start = st.date_input("Start Date", value=date.today() - timedelta(days=365))
        with col_e:
            custom_end = st.date_input("End Date", value=date.today())

    with st.spinner("Loading historical data..."):
        if period_option == "Custom" and custom_start and custom_end:
            df = get_custom_range_data(
                ticker,
                start=str(custom_start),
                end=str(custom_end),
                interval=interval,
            )
        else:
            df = get_historical_data(ticker, period=period_option, interval=interval)

    if df.empty:
        st.error("No data found. Try a different ticker or date range.")
        return

    # ── Main Price Chart ───────────────────────────────────────────────────
    st.markdown(f"### {ticker} — {period_option} {interval.upper()} Chart")

    if chart_type == "Candlestick":
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color="#00c853", decreasing_line_color="#ff1744",
        )])
    elif chart_type == "OHLC":
        fig = go.Figure(data=[go.Ohlc(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
        )])
    elif chart_type == "Line":
        fig = go.Figure(data=[go.Scatter(
            x=df.index, y=df["Close"], mode="lines",
            line=dict(color="#00d4ff", width=2),
        )])
    else:  # Area
        fig = go.Figure(data=[go.Scatter(
            x=df.index, y=df["Close"], mode="lines",
            fill="tozeroy", fillcolor="rgba(0,212,255,0.1)",
            line=dict(color="#00d4ff", width=2),
        )])

    fig.update_layout(
        template=CHART_THEME, height=500,
        xaxis_rangeslider_visible=(chart_type in ["Candlestick", "OHLC"]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Volume Chart ───────────────────────────────────────────────────────
    with st.expander("📊 Volume Analysis", expanded=True):
        colors = ["#00c853" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ff1744"
                  for i in range(len(df))]
        vol_fig = go.Figure(data=[go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=colors, opacity=0.8, name="Volume",
        )])
        # Volume moving average
        vol_ma = df["Volume"].rolling(20).mean()
        vol_fig.add_trace(go.Scatter(
            x=df.index, y=vol_ma, name="Vol MA(20)",
            line=dict(color="#ff9800", width=2),
        ))
        vol_fig.update_layout(
            title="Trading Volume", template=CHART_THEME, height=300,
        )
        st.plotly_chart(vol_fig, use_container_width=True)

    # ── Performance Statistics ─────────────────────────────────────────────
    st.markdown("### 📈 Performance Statistics")
    returns = df["Close"].pct_change().dropna()
    total_return = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100
    annualized_vol = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    max_drawdown = ((df["Close"] / df["Close"].cummax()) - 1).min() * 100
    best_day = returns.max() * 100
    worst_day = returns.min() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{total_return:+.2f}%")
    c2.metric("Annualized Volatility", f"{annualized_vol:.2f}%")
    c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c4.metric("Max Drawdown", f"{max_drawdown:.2f}%")

    c1b, c2b, c3b, c4b = st.columns(4)
    c1b.metric("Best Day", f"{best_day:+.2f}%")
    c2b.metric("Worst Day", f"{worst_day:+.2f}%")
    c3b.metric("Avg Daily Return", f"{returns.mean() * 100:+.4f}%")
    c4b.metric("Positive Days", f"{(returns > 0).sum()} / {len(returns)}")

    # ── Return Distribution ────────────────────────────────────────────────
    col_dist, col_cum = st.columns(2)

    with col_dist:
        hist_fig = go.Figure(data=[go.Histogram(
            x=returns * 100, nbinsx=50,
            marker_color="#7b2ff7", opacity=0.8,
        )])
        hist_fig.update_layout(
            title="Daily Return Distribution",
            template=CHART_THEME, height=350,
            xaxis_title="Daily Return (%)", yaxis_title="Frequency",
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    with col_cum:
        cumulative = (1 + returns).cumprod() * 100 - 100
        cum_fig = go.Figure(data=[go.Scatter(
            x=cumulative.index, y=cumulative,
            mode="lines", line=dict(color="#00d4ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
        )])
        cum_fig.update_layout(
            title="Cumulative Return (%)",
            template=CHART_THEME, height=350,
        )
        st.plotly_chart(cum_fig, use_container_width=True)

    # ── OHLC Data Table ────────────────────────────────────────────────────
    with st.expander("🗂️ Raw OHLCV Data"):
        display_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        display_df.index = display_df.index.strftime("%Y-%m-%d")
        display_df = display_df.sort_index(ascending=False)
        for col in ["Open", "High", "Low", "Close"]:
            display_df[col] = display_df[col].round(2)
        st.dataframe(display_df, use_container_width=True, height=400)

        csv = display_df.to_csv().encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name=f"{ticker}_historical_{period_option}.csv",
            mime="text/csv",
        )
