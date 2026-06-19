"""
AI Predictions page — Linear Regression, Random Forest, LSTM forecasts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_fetcher import get_historical_data, get_stock_info, get_current_price
from utils.ml_models import (
    linear_regression_predict, random_forest_predict, lstm_predict, generate_recommendation
)
from utils.technical_indicators import get_trading_signals
from utils.charts import prediction_chart, CHART_THEME


def render(ticker: str):
    st.markdown('<h1 class="main-header">🤖 AI Price Predictions</h1>', unsafe_allow_html=True)

    if not ticker:
        st.warning("Enter a ticker symbol in the sidebar.")
        return

    st.info("⚠️ Predictions are for educational purposes only. Not financial advice.")

    # ── Settings ───────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        forecast_days = st.slider("Forecast Days", 7, 90, 30)
    with col2:
        period = st.selectbox("Training Data", ["1y", "2y", "5y"], index=1)
    with col3:
        model_choice = st.selectbox(
            "Model",
            ["Linear Regression", "Random Forest", "LSTM (Deep Learning)", "All Models"],
        )

    with st.spinner("Loading data..."):
        df = get_historical_data(ticker, period=period)
        info = get_stock_info(ticker)
        price_data = get_current_price(ticker)

    if df.empty:
        st.error("No data found.")
        return

    current_price = price_data.get("price", df["Close"].iloc[-1]) if price_data else df["Close"].iloc[-1]
    pe_ratio = info.get("trailingPE") if info else None

    # ── Run Models ─────────────────────────────────────────────────────────
    run_lr = model_choice in ["Linear Regression", "All Models"]
    run_rf = model_choice in ["Random Forest", "All Models"]
    run_lstm = model_choice in ["LSTM (Deep Learning)", "All Models"]

    results = {}
    error_msgs = []

    if run_lr:
        with st.spinner("Running Linear Regression..."):
            lr_result = linear_regression_predict(df, forecast_days=forecast_days)
            if "error" in lr_result:
                error_msgs.append(f"Linear Regression: {lr_result['error']}")
            else:
                results["Linear Regression"] = lr_result

    if run_rf:
        with st.spinner("Running Random Forest..."):
            rf_result = random_forest_predict(df, forecast_days=forecast_days)
            if "error" in rf_result:
                error_msgs.append(f"Random Forest: {rf_result['error']}")
            else:
                results["Random Forest"] = rf_result

    if run_lstm:
        with st.spinner("Training LSTM model... (this may take 1-2 minutes)"):
            lstm_result = lstm_predict(df, forecast_days=forecast_days)
            if "error" in lstm_result:
                error_msgs.append(f"LSTM: {lstm_result['error']}")
            else:
                results["LSTM (Deep Learning)"] = lstm_result

    for msg in error_msgs:
        st.warning(msg)

    if not results:
        st.error("No models completed successfully.")
        return

    # ── Overall Recommendation Banner ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 AI Trading Recommendation")

    # Use the best model (prefer RF > LSTM > LR for recommendation)
    best_model_key = (
        "Random Forest" if "Random Forest" in results else
        "LSTM (Deep Learning)" if "LSTM (Deep Learning)" in results else
        "Linear Regression"
    )
    best_result = results[best_model_key]
    predicted_price = best_result["future_prices"][-1] if best_result["future_prices"] else current_price

    tech_signals = get_trading_signals(df)
    recommendation = generate_recommendation(
        current_price=current_price,
        predicted_price=predicted_price,
        technical_score=tech_signals["score"],
        pe_ratio=pe_ratio,
    )

    rec_color = {
        "STRONG BUY": "#00c853", "BUY": "#8bc34a",
        "HOLD": "#ffd600", "SELL": "#ff9800", "STRONG SELL": "#ff1744",
    }.get(recommendation["recommendation"], "#aaa")

    col_rec, col_details = st.columns([1, 2])

    with col_rec:
        st.markdown(f"""
        <div style="background:#1e1e2e;border-radius:16px;padding:2rem;border:3px solid {rec_color};text-align:center;">
            <div style="color:#aaa;margin-bottom:0.5rem;">AI Recommendation</div>
            <div style="color:{rec_color};font-size:2.5rem;font-weight:800;">{recommendation['recommendation']}</div>
            <div style="margin-top:1rem;">
                <div style="color:#aaa;font-size:0.85rem;">Confidence</div>
                <div style="color:white;font-size:1.5rem;">{recommendation['confidence']}%</div>
            </div>
            <div style="margin-top:0.5rem;">
                <div style="color:#aaa;font-size:0.85rem;">Expected Move</div>
                <div style="color:{rec_color};font-size:1.2rem;">{recommendation['price_upside']:+.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_details:
        st.markdown("**Reasoning:**")
        for reason in recommendation["reasons"]:
            icon = "🟢" if "Buy" in reason or "bullish" in reason.lower() else (
                "🔴" if "Sell" in reason or "bearish" in reason.lower() else "🟡"
            )
            st.markdown(f"{icon} {reason}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"${current_price:.2f}")
        c2.metric("Target Price", f"${predicted_price:.2f}")
        c3.metric("Model Used", best_model_key.replace(" (Deep Learning)", ""))

    st.markdown("---")

    # ── Individual Model Results ───────────────────────────────────────────
    for model_name, result in results.items():
        st.markdown(f"### 📈 {model_name}")

        # Metrics
        metrics = result.get("metrics", {})
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("MAE", f"${metrics.get('MAE', 'N/A')}")
        col_m2.metric("RMSE", f"${metrics.get('RMSE', 'N/A')}")
        col_m3.metric("R² Score", f"{metrics.get('R2', 'N/A')}")
        col_m4.metric("MAPE", f"{metrics.get('MAPE', 'N/A')}%")

        # Forecast chart
        test_size = result.get("test_size", 0)
        test_dates = df.index[-test_size:] if test_size > 0 else []
        historical_dates = df.index[:-test_size] if test_size > 0 else df.index

        fig = prediction_chart(
            historical_dates=historical_dates,
            historical_prices=df["Close"][:-test_size].tolist() if test_size > 0 else df["Close"].tolist(),
            test_dates=test_dates,
            test_actual=result.get("test_actual", []),
            test_predicted=result.get("test_predicted", []),
            future_dates=result.get("future_dates", []),
            future_prices=result.get("future_prices", []),
            model_name=f"{ticker} — {model_name}",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Future price table
        with st.expander(f"📋 {model_name} — Forecast Table"):
            future_df = pd.DataFrame({
                "Date": [str(d)[:10] for d in result["future_dates"]],
                "Predicted Price": [f"${p:.2f}" for p in result["future_prices"]],
                "Change from Today": [
                    f"{((p - current_price) / current_price) * 100:+.2f}%"
                    for p in result["future_prices"]
                ],
            })
            st.dataframe(future_df, hide_index=True, use_container_width=True)

        # Feature importance (RF only)
        if "feature_importance" in result:
            with st.expander("🔍 Feature Importance"):
                feat_df = pd.DataFrame(result["feature_importance"], columns=["Feature", "Importance"])
                fig_fi = go.Figure(go.Bar(
                    x=feat_df["Importance"], y=feat_df["Feature"],
                    orientation="h",
                    marker_color="#7b2ff7",
                ))
                fig_fi.update_layout(template=CHART_THEME, height=350, title="Top Features")
                st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("---")

    # ── Model Comparison (if All Models) ──────────────────────────────────
    if len(results) > 1 and "future_prices" in list(results.values())[0]:
        st.markdown("### ⚖️ Model Forecast Comparison")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=df.index[-60:], y=df["Close"].iloc[-60:],
            name="Historical", line=dict(color="#00d4ff", width=2),
        ))
        colors_map = {"Linear Regression": "#ff9800", "Random Forest": "#00c853", "LSTM (Deep Learning)": "#e91e63"}
        for model_name, result in results.items():
            fig_cmp.add_trace(go.Scatter(
                x=result["future_dates"], y=result["future_prices"],
                name=model_name, mode="lines+markers",
                line=dict(color=colors_map.get(model_name, "#fff"), width=2, dash="dash"),
                marker=dict(size=4),
            ))
        fig_cmp.update_layout(
            title=f"{ticker} — Model Forecast Comparison",
            template=CHART_THEME, height=450,
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Summary table
        summary_rows = []
        for model_name, result in results.items():
            fps = result["future_prices"]
            summary_rows.append({
                "Model": model_name,
                "7-Day Target": f"${fps[6]:.2f}" if len(fps) > 6 else "N/A",
                "30-Day Target": f"${fps[29]:.2f}" if len(fps) > 29 else f"${fps[-1]:.2f}",
                "Expected Move": f"{((fps[-1] - current_price) / current_price) * 100:+.1f}%",
                "R² Score": result["metrics"].get("R2", "N/A"),
            })
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
