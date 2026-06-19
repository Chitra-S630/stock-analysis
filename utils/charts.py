"""
Reusable Plotly chart builders.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


CHART_THEME = "plotly_dark"
COLORS = {
    "price": "#00d4ff",
    "volume": "#7b2ff7",
    "sma20": "#f9a825",
    "sma50": "#e91e63",
    "sma200": "#4caf50",
    "ema12": "#ff9800",
    "ema26": "#03a9f4",
    "macd": "#00e5ff",
    "signal": "#ff4081",
    "histogram_pos": "#00c853",
    "histogram_neg": "#ff1744",
    "rsi": "#7c4dff",
    "bb_upper": "#ff6d00",
    "bb_lower": "#ff6d00",
    "bb_middle": "#ffd600",
    "positive": "#00c853",
    "negative": "#ff1744",
}


def candlestick_chart(df: pd.DataFrame, ticker: str, indicators: dict = None) -> go.Figure:
    """Full candlestick chart with optional overlays."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color=COLORS["positive"],
            decreasing_line_color=COLORS["negative"],
        ),
        row=1, col=1,
    )

    # Overlay indicators
    if indicators:
        if "sma20" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=indicators["sma20"], name="SMA 20",
                line=dict(color=COLORS["sma20"], width=1.5)), row=1, col=1)
        if "sma50" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=indicators["sma50"], name="SMA 50",
                line=dict(color=COLORS["sma50"], width=1.5)), row=1, col=1)
        if "sma200" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=indicators["sma200"], name="SMA 200",
                line=dict(color=COLORS["sma200"], width=1.5)), row=1, col=1)
        if "ema20" in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=indicators["ema20"], name="EMA 20",
                line=dict(color=COLORS["ema12"], width=1.5, dash="dot")), row=1, col=1)
        if "bb" in indicators:
            bb = indicators["bb"]
            fig.add_trace(go.Scatter(x=df.index, y=bb["BB_Upper"], name="BB Upper",
                line=dict(color=COLORS["bb_upper"], width=1, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bb["BB_Lower"], name="BB Lower",
                line=dict(color=COLORS["bb_lower"], width=1, dash="dash"),
                fill="tonexty", fillcolor="rgba(255,109,0,0.05)"), row=1, col=1)

    # Volume
    colors = [COLORS["positive"] if df["Close"].iloc[i] >= df["Open"].iloc[i]
              else COLORS["negative"] for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume",
               marker_color=colors, opacity=0.7),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"{ticker} Price Chart",
        template=CHART_THEME,
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def line_chart(df: pd.DataFrame, y_col: str, title: str, color: str = None) -> go.Figure:
    """Simple line chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df[y_col], mode="lines",
        line=dict(color=color or COLORS["price"], width=2),
        fill="tozeroy", fillcolor=f"rgba(0,212,255,0.05)",
        name=y_col,
    ))
    fig.update_layout(title=title, template=CHART_THEME, height=400)
    return fig


def rsi_chart(rsi: pd.Series) -> go.Figure:
    """RSI chart with overbought/oversold zones."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rsi.index, y=rsi.values, mode="lines",
        line=dict(color=COLORS["rsi"], width=2), name="RSI",
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.05)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.05)
    fig.update_layout(
        title="RSI (14)", template=CHART_THEME, height=300,
        yaxis=dict(range=[0, 100]),
    )
    return fig


def macd_chart(macd_df: pd.DataFrame) -> go.Figure:
    """MACD chart with histogram."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=macd_df.index, y=macd_df["MACD"],
        name="MACD", line=dict(color=COLORS["macd"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=macd_df.index, y=macd_df["Signal"],
        name="Signal", line=dict(color=COLORS["signal"], width=1.5),
    ))
    colors = [COLORS["histogram_pos"] if v >= 0 else COLORS["histogram_neg"]
              for v in macd_df["Histogram"]]
    fig.add_trace(go.Bar(
        x=macd_df.index, y=macd_df["Histogram"],
        name="Histogram", marker_color=colors, opacity=0.7,
    ))
    fig.update_layout(title="MACD", template=CHART_THEME, height=300)
    return fig


def stochastic_chart(stoch_df: pd.DataFrame) -> go.Figure:
    """Stochastic Oscillator chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stoch_df.index, y=stoch_df["%K"],
        name="%K", line=dict(color=COLORS["macd"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=stoch_df.index, y=stoch_df["%D"],
        name="%D", line=dict(color=COLORS["signal"], width=1.5),
    ))
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought (80)")
    fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold (20)")
    fig.update_layout(
        title="Stochastic Oscillator", template=CHART_THEME, height=300,
        yaxis=dict(range=[0, 100]),
    )
    return fig


def prediction_chart(
    historical_dates, historical_prices,
    test_dates, test_actual, test_predicted,
    future_dates, future_prices,
    model_name: str,
) -> go.Figure:
    """Chart showing historical prices, test prediction, and future forecast."""
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=historical_dates, y=historical_prices,
        name="Historical", mode="lines",
        line=dict(color=COLORS["price"], width=1.5),
    ))

    # Test predictions
    fig.add_trace(go.Scatter(
        x=test_dates, y=test_actual,
        name="Actual (Test)", mode="lines",
        line=dict(color=COLORS["positive"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=test_dates, y=test_predicted,
        name="Predicted (Test)", mode="lines",
        line=dict(color=COLORS["signal"], width=2, dash="dot"),
    ))

    # Future forecast
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_prices,
        name="Forecast", mode="lines+markers",
        line=dict(color="#ff9800", width=2.5, dash="dash"),
        marker=dict(size=4),
    ))

    # Confidence zone
    future_std = np.std(future_prices) * 0.5 if len(future_prices) > 1 else 0
    upper_bound = [p + future_std * (1 + i * 0.05) for i, p in enumerate(future_prices)]
    lower_bound = [p - future_std * (1 + i * 0.05) for i, p in enumerate(future_prices)]

    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=upper_bound + lower_bound[::-1],
        fill="toself", fillcolor="rgba(255,152,0,0.1)",
        line=dict(color="rgba(255,152,0,0)"),
        name="Forecast Confidence",
    ))

    fig.update_layout(
        title=f"{model_name} — Price Prediction",
        template=CHART_THEME,
        height=500,
        showlegend=True,
    )
    return fig


def performance_comparison_chart(returns_df: pd.DataFrame) -> go.Figure:
    """Normalized performance comparison of multiple stocks."""
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, col in enumerate(returns_df.columns):
        fig.add_trace(go.Scatter(
            x=returns_df.index, y=returns_df[col],
            name=col, mode="lines",
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Performance Comparison (Normalized to 100)",
        template=CHART_THEME, height=450,
        yaxis_title="Normalized Price",
    )
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Correlation heatmap for multiple stocks."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title="Return Correlation Matrix",
        template=CHART_THEME, height=450,
    )
    return fig


def sentiment_gauge(score: float) -> go.Figure:
    """Gauge chart showing overall sentiment score."""
    gauge_val = (score + 1) / 2 * 100  # map [-1,1] → [0,100]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_val,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Sentiment Score"},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#7b2ff7"},
            "steps": [
                {"range": [0, 30], "color": "#ff1744"},
                {"range": [30, 45], "color": "#ff9800"},
                {"range": [45, 55], "color": "#ffd600"},
                {"range": [55, 70], "color": "#8bc34a"},
                {"range": [70, 100], "color": "#00c853"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": gauge_val,
            },
        },
    ))
    fig.update_layout(template=CHART_THEME, height=300)
    return fig
