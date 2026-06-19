"""
Stock Comparison page — side-by-side metrics, performance, correlation, risk.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.data_fetcher import get_stock_info, get_multiple_stocks, get_historical_data
from utils.charts import performance_comparison_chart, correlation_heatmap, CHART_THEME


def render():
    st.markdown('<h1 class="main-header">⚖️ Stock Comparison</h1>', unsafe_allow_html=True)

    # ── Ticker Input ───────────────────────────────────────────────────────
    st.markdown("### Select Stocks to Compare")
    default_tickers = "AAPL, MSFT, GOOGL, AMZN"
    ticker_input = st.text_input(
        "Enter up to 6 ticker symbols (comma-separated)",
        value=default_tickers,
    )
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()][:6]

    period = st.selectbox("Comparison Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

    if len(tickers) < 2:
        st.warning("Please enter at least 2 ticker symbols.")
        return

    with st.spinner("Loading comparison data..."):
        price_data = get_multiple_stocks(tickers, period=period)
        infos = {t: get_stock_info(t) for t in tickers}

    valid_tickers = [t for t in tickers if t in price_data and not price_data[t].empty]
    if len(valid_tickers) < 2:
        st.error("Could not load data for enough tickers.")
        return

    # ── Performance Comparison Chart ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Performance Comparison")

    # Build normalized dataframe
    combined = pd.DataFrame({t: price_data[t] for t in valid_tickers}).dropna()
    normalized = (combined / combined.iloc[0]) * 100

    st.plotly_chart(performance_comparison_chart(normalized), use_container_width=True)

    # ── Return Summary ─────────────────────────────────────────────────────
    st.markdown("### 📊 Return Summary")
    return_data = []
    for t in valid_tickers:
        prices = price_data[t].dropna()
        if len(prices) > 1:
            total_ret = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100
            daily_ret = prices.pct_change().dropna()
            volatility = daily_ret.std() * np.sqrt(252) * 100
            sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
            max_dd = ((prices / prices.cummax()) - 1).min() * 100
            return_data.append({
                "Ticker": t,
                "Total Return": f"{total_ret:+.2f}%",
                "Annual Volatility": f"{volatility:.2f}%",
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Max Drawdown": f"{max_dd:.2f}%",
                "Current Price": f"${prices.iloc[-1]:.2f}",
            })

    st.dataframe(pd.DataFrame(return_data), hide_index=True, use_container_width=True)

    # ── Correlation Heatmap ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔗 Correlation Matrix")
    returns_df = combined.pct_change().dropna()
    corr = returns_df.corr()
    st.plotly_chart(correlation_heatmap(corr), use_container_width=True)

    # ── Risk vs Return Scatter ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚠️ Risk vs Return")
    risk_return = []
    for t in valid_tickers:
        prices = price_data[t].dropna()
        if len(prices) > 1:
            daily_ret = prices.pct_change().dropna()
            annual_ret = daily_ret.mean() * 252 * 100
            annual_vol = daily_ret.std() * np.sqrt(252) * 100
            risk_return.append({"Ticker": t, "Annual Return (%)": annual_ret, "Annual Volatility (%)": annual_vol})

    if risk_return:
        rr_df = pd.DataFrame(risk_return)
        fig_rr = px.scatter(
            rr_df, x="Annual Volatility (%)", y="Annual Return (%)",
            text="Ticker", size_max=20,
            template=CHART_THEME,
            title="Risk vs Return (Annualized)",
            color="Ticker",
        )
        fig_rr.update_traces(textposition="top center", marker=dict(size=14))
        fig_rr.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_rr.update_layout(height=450)
        st.plotly_chart(fig_rr, use_container_width=True)

    # ── Side-by-Side Financial Metrics ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💼 Side-by-Side Financial Metrics")
    metrics_rows = []
    metric_keys = [
        ("Market Cap", "marketCap", "cap"),
        ("P/E Ratio", "trailingPE", "ratio"),
        ("Forward P/E", "forwardPE", "ratio"),
        ("P/B Ratio", "priceToBook", "ratio"),
        ("EPS (TTM)", "trailingEps", "dollar"),
        ("Dividend Yield", "dividendYield", "pct"),
        ("Debt/Equity", "debtToEquity", "ratio"),
        ("Net Margin", "profitMargins", "pct"),
        ("ROE", "returnOnEquity", "pct"),
        ("Beta", "beta", "ratio"),
        ("52W High", "fiftyTwoWeekHigh", "dollar"),
        ("52W Low", "fiftyTwoWeekLow", "dollar"),
    ]

    for label, key, fmt in metric_keys:
        row = {"Metric": label}
        for t in valid_tickers:
            val = infos[t].get(key)
            if val is None:
                row[t] = "N/A"
            elif fmt == "cap":
                if val >= 1e12:
                    row[t] = f"${val/1e12:.2f}T"
                elif val >= 1e9:
                    row[t] = f"${val/1e9:.2f}B"
                else:
                    row[t] = f"${val/1e6:.2f}M"
            elif fmt == "pct":
                row[t] = f"{val * 100:.2f}%"
            elif fmt == "dollar":
                row[t] = f"${val:.2f}"
            else:
                row[t] = f"{val:.2f}"
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    # ── Volume Comparison ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📦 Volume Comparison (Most Recent)")
    vol_data = []
    for t in valid_tickers:
        df = get_historical_data(t, period="5d")
        if not df.empty:
            avg_vol = df["Volume"].mean()
            vol_data.append({"Ticker": t, "Avg Daily Volume": avg_vol})

    if vol_data:
        vol_df = pd.DataFrame(vol_data).sort_values("Avg Daily Volume", ascending=False)
        fig_vol = go.Figure(go.Bar(
            x=vol_df["Ticker"], y=vol_df["Avg Daily Volume"],
            marker_color=px.colors.qualitative.Plotly[:len(vol_df)],
            text=[f"{v/1e6:.1f}M" for v in vol_df["Avg Daily Volume"]],
            textposition="auto",
        ))
        fig_vol.update_layout(
            title="Average Daily Volume (5-day)", template=CHART_THEME, height=350,
        )
        st.plotly_chart(fig_vol, use_container_width=True)
