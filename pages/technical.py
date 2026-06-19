"""
Technical Analysis page — all indicators, signals, support/resistance.
"""

import streamlit as st
import pandas as pd
from utils.data_fetcher import get_historical_data
from utils.technical_indicators import (
    calculate_sma, calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_stochastic, calculate_atr,
    calculate_obv, calculate_vwap, detect_support_resistance,
    detect_trend, get_trading_signals,
)
from utils.charts import (
    candlestick_chart, rsi_chart, macd_chart, stochastic_chart, CHART_THEME
)
import plotly.graph_objects as go


def render(ticker: str):
    st.markdown('<h1 class="main-header">🔬 Technical Analysis</h1>', unsafe_allow_html=True)

    if not ticker:
        st.warning("Enter a ticker symbol in the sidebar.")
        return

    # Controls
    col1, col2 = st.columns([3, 1])
    with col1:
        period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1)
    with col2:
        interval = st.selectbox("Interval", ["1d", "1wk"])

    with st.spinner("Calculating indicators..."):
        df = get_historical_data(ticker, period=period, interval=interval)

    if df.empty:
        st.error("No data found.")
        return

    # ── Overall Signal ─────────────────────────────────────────────────────
    signals = get_trading_signals(df)
    trend = detect_trend(df)

    sig_col = {
        "STRONG BUY": "#00c853", "BUY": "#8bc34a",
        "HOLD": "#ffd600", "SELL": "#ff9800", "STRONG SELL": "#ff1744",
    }.get(signals["signal"], "#aaa")

    st.markdown(f"""
    <div style="display:flex;gap:1rem;margin-bottom:1rem;">
        <div style="background:#1e1e2e;border-radius:12px;padding:1rem 2rem;border:2px solid {sig_col};text-align:center;flex:1">
            <div style="color:#aaa;font-size:0.9rem;">Overall Signal</div>
            <div style="color:{sig_col};font-size:2.2rem;font-weight:700;">{signals['signal']}</div>
        </div>
        <div style="background:#1e1e2e;border-radius:12px;padding:1rem 2rem;border:1px solid #333;text-align:center;flex:1">
            <div style="color:#aaa;font-size:0.9rem;">Trend</div>
            <div style="font-size:1.5rem;font-weight:700;">{trend}</div>
        </div>
        <div style="background:#1e1e2e;border-radius:12px;padding:1rem 2rem;border:1px solid #333;text-align:center;flex:1">
            <div style="color:#aaa;font-size:0.9rem;">Signal Score</div>
            <div style="color:{sig_col};font-size:2.2rem;font-weight:700;">{signals['score']:+d}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Signal reasoning
    with st.expander("📋 Signal Breakdown"):
        for r in signals["reasons"]:
            icon = "🟢" if "Bullish" in r else ("🔴" if "Bearish" in r else "🟡")
            st.markdown(f"{icon} {r}")

    st.markdown("---")

    # ── Tabs for each indicator group ─────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Moving Averages", "RSI", "MACD", "Bollinger Bands", "Stochastic & More"
    ])

    with tab1:
        st.markdown("### Moving Averages")
        col_s, col_e = st.columns(2)
        with col_s:
            sma_periods = st.multiselect("SMA Periods", [10, 20, 50, 100, 200],
                                         default=[20, 50, 200])
        with col_e:
            ema_periods = st.multiselect("EMA Periods", [9, 12, 20, 26, 50],
                                         default=[12, 26])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                                  line=dict(color="#00d4ff", width=2)))

        colors = ["#f9a825", "#e91e63", "#4caf50", "#ff9800", "#00bcd4"]
        for i, p in enumerate(sma_periods):
            sma = calculate_sma(df, p)
            fig.add_trace(go.Scatter(x=df.index, y=sma, name=f"SMA {p}",
                                      line=dict(color=colors[i % len(colors)], width=1.5)))

        ema_colors = ["#ff5722", "#9c27b0", "#2196f3"]
        for i, p in enumerate(ema_periods):
            ema = calculate_ema(df, p)
            fig.add_trace(go.Scatter(x=df.index, y=ema, name=f"EMA {p}",
                                      line=dict(color=ema_colors[i % len(ema_colors)],
                                                width=1.5, dash="dot")))

        fig.update_layout(title=f"{ticker} — Moving Averages", template=CHART_THEME, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # MA values table
        ma_data = {"Metric": [], "Value": [], "Status": []}
        current = df["Close"].iloc[-1]
        for p in sma_periods:
            val = calculate_sma(df, p).iloc[-1]
            status = "🟢 Above" if current > val else "🔴 Below"
            ma_data["Metric"].append(f"SMA {p}")
            ma_data["Value"].append(f"${val:.2f}")
            ma_data["Status"].append(status)
        for p in ema_periods:
            val = calculate_ema(df, p).iloc[-1]
            status = "🟢 Above" if current > val else "🔴 Below"
            ma_data["Metric"].append(f"EMA {p}")
            ma_data["Value"].append(f"${val:.2f}")
            ma_data["Status"].append(status)
        st.dataframe(pd.DataFrame(ma_data), hide_index=True, use_container_width=True)

    with tab2:
        st.markdown("### Relative Strength Index (RSI)")
        rsi_period = st.slider("RSI Period", 7, 28, 14)
        rsi = calculate_rsi(df, window=rsi_period)
        last_rsi = rsi.iloc[-1]

        c1, c2 = st.columns([1, 3])
        with c1:
            rsi_color = "#ff1744" if last_rsi > 70 else ("#00c853" if last_rsi < 30 else "#ffd600")
            rsi_label = "Overbought 🔴" if last_rsi > 70 else ("Oversold 🟢" if last_rsi < 30 else "Neutral 🟡")
            st.markdown(f"""
            <div style="background:#1e1e2e;border-radius:12px;padding:1.5rem;border:2px solid {rsi_color};text-align:center;">
                <div style="color:#aaa;">Current RSI</div>
                <div style="color:{rsi_color};font-size:2.5rem;font-weight:700;">{last_rsi:.1f}</div>
                <div style="color:{rsi_color};">{rsi_label}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.plotly_chart(rsi_chart(rsi), use_container_width=True)

    with tab3:
        st.markdown("### MACD")
        c1, c2, c3 = st.columns(3)
        fast = c1.number_input("Fast EMA", value=12, min_value=2, max_value=50)
        slow = c2.number_input("Slow EMA", value=26, min_value=5, max_value=100)
        signal_p = c3.number_input("Signal", value=9, min_value=2, max_value=30)

        macd_df = calculate_macd(df, fast=int(fast), slow=int(slow), signal=int(signal_p))
        last_macd = macd_df["MACD"].iloc[-1]
        last_signal = macd_df["Signal"].iloc[-1]
        last_hist = macd_df["Histogram"].iloc[-1]

        c1, c2, c3 = st.columns(3)
        c1.metric("MACD Line", f"{last_macd:.4f}")
        c2.metric("Signal Line", f"{last_signal:.4f}")
        c3.metric("Histogram", f"{last_hist:.4f}",
                  delta="Bullish" if last_hist > 0 else "Bearish")

        st.plotly_chart(macd_chart(macd_df), use_container_width=True)

    with tab4:
        st.markdown("### Bollinger Bands")
        bb_period = st.slider("BB Period", 10, 50, 20)
        bb_std = st.slider("Std Dev Multiplier", 1.0, 3.0, 2.0, step=0.5)

        bb = calculate_bollinger_bands(df, window=bb_period, std_dev=bb_std)
        current_price = df["Close"].iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Upper Band", f"${bb['BB_Upper'].iloc[-1]:.2f}")
        c2.metric("Middle (SMA)", f"${bb['BB_Middle'].iloc[-1]:.2f}")
        c3.metric("Lower Band", f"${bb['BB_Lower'].iloc[-1]:.2f}")
        bw = (bb["BB_Upper"].iloc[-1] - bb["BB_Lower"].iloc[-1]) / bb["BB_Middle"].iloc[-1] * 100
        c4.metric("Band Width %", f"{bw:.2f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price",
                                  line=dict(color="#00d4ff", width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=bb["BB_Upper"], name="Upper",
                                  line=dict(color="#ff6d00", width=1.5, dash="dash")))
        fig.add_trace(go.Scatter(x=df.index, y=bb["BB_Middle"], name="Middle",
                                  line=dict(color="#ffd600", width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=bb["BB_Lower"], name="Lower",
                                  line=dict(color="#ff6d00", width=1.5, dash="dash"),
                                  fill="tonexty", fillcolor="rgba(255,109,0,0.05)"))
        fig.update_layout(title=f"{ticker} — Bollinger Bands ({bb_period},{bb_std})",
                           template=CHART_THEME, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("### Stochastic Oscillator")
            stoch = calculate_stochastic(df)
            st.plotly_chart(stochastic_chart(stoch), use_container_width=True)

            # ATR
            st.markdown("### Average True Range (ATR)")
            atr = calculate_atr(df)
            fig_atr = go.Figure(go.Scatter(x=df.index, y=atr, name="ATR",
                                            line=dict(color="#7c4dff", width=2)))
            fig_atr.update_layout(title="ATR (14)", template=CHART_THEME, height=300)
            st.plotly_chart(fig_atr, use_container_width=True)

        with col_r:
            st.markdown("### On-Balance Volume (OBV)")
            obv = calculate_obv(df)
            fig_obv = go.Figure(go.Scatter(x=df.index, y=obv, name="OBV",
                                            fill="tozeroy", fillcolor="rgba(123,47,247,0.1)",
                                            line=dict(color="#7b2ff7", width=2)))
            fig_obv.update_layout(title="OBV", template=CHART_THEME, height=300)
            st.plotly_chart(fig_obv, use_container_width=True)

            # Support & Resistance
            st.markdown("### Support & Resistance Levels")
            sr = detect_support_resistance(df)
            current = df["Close"].iloc[-1]

            st.markdown("**Resistance Levels**")
            for level in sr["resistance"]:
                dist = ((level - current) / current) * 100
                st.markdown(f"🔴 ${level:.2f} ({dist:+.1f}% from current)")

            st.markdown("**Support Levels**")
            for level in sr["support"]:
                dist = ((level - current) / current) * 100
                st.markdown(f"🟢 ${level:.2f} ({dist:+.1f}% from current)")
