"""
Technical analysis indicator calculations.
"""

import pandas as pd
import numpy as np


def calculate_sma(df: pd.DataFrame, window: int, column: str = "Close") -> pd.Series:
    """Simple Moving Average."""
    return df[column].rolling(window=window).mean()


def calculate_ema(df: pd.DataFrame, window: int, column: str = "Close") -> pd.Series:
    """Exponential Moving Average."""
    return df[column].ewm(span=window, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, window: int = 14, column: str = "Close") -> pd.Series:
    """Relative Strength Index."""
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "Close",
) -> pd.DataFrame:
    """MACD, Signal Line, and Histogram."""
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line, "Histogram": histogram})


def calculate_bollinger_bands(
    df: pd.DataFrame, window: int = 20, std_dev: float = 2.0, column: str = "Close"
) -> pd.DataFrame:
    """Bollinger Bands: Upper, Middle (SMA), Lower."""
    sma = df[column].rolling(window=window).mean()
    std = df[column].rolling(window=window).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return pd.DataFrame({"BB_Upper": upper, "BB_Middle": sma, "BB_Lower": lower})


def calculate_stochastic(
    df: pd.DataFrame, k_window: int = 14, d_window: int = 3
) -> pd.DataFrame:
    """Stochastic Oscillator %K and %D."""
    low_min = df["Low"].rolling(window=k_window).min()
    high_max = df["High"].rolling(window=k_window).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_window).mean()
    return pd.DataFrame({"%K": k, "%D": d})


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price (intraday-style, reset daily)."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cumulative_tp_vol = (typical_price * df["Volume"]).cumsum()
    cumulative_vol = df["Volume"].cumsum()
    return cumulative_tp_vol / cumulative_vol


def detect_support_resistance(df: pd.DataFrame, window: int = 20, n_levels: int = 5) -> dict:
    """Detect key support and resistance levels using local extrema."""
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    resistance_levels = []
    support_levels = []

    for i in range(window, len(df) - window):
        # Local high → potential resistance
        if highs[i] == max(highs[i - window : i + window]):
            resistance_levels.append(highs[i])
        # Local low → potential support
        if lows[i] == min(lows[i - window : i + window]):
            support_levels.append(lows[i])

    # Cluster nearby levels
    def cluster_levels(levels, tolerance=0.02):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        clusters.append(np.mean(current_cluster))
        return clusters

    resistance = sorted(cluster_levels(resistance_levels), reverse=True)[:n_levels]
    support = sorted(cluster_levels(support_levels))[:n_levels]

    return {"resistance": resistance, "support": support}


def detect_trend(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> str:
    """Detect overall price trend."""
    if len(df) < long_window:
        return "Insufficient data"
    sma_short = df["Close"].rolling(short_window).mean().iloc[-1]
    sma_long = df["Close"].rolling(long_window).mean().iloc[-1]
    current_price = df["Close"].iloc[-1]

    if current_price > sma_short > sma_long:
        return "Strong Uptrend 📈"
    elif current_price > sma_long:
        return "Moderate Uptrend 📈"
    elif current_price < sma_short < sma_long:
        return "Strong Downtrend 📉"
    elif current_price < sma_long:
        return "Moderate Downtrend 📉"
    else:
        return "Sideways / Neutral ↔️"


def get_trading_signals(df: pd.DataFrame) -> dict:
    """Generate buy/sell signals from multiple indicators."""
    signals = {}
    reasons = []
    score = 0  # Positive = bullish, negative = bearish

    if len(df) < 50:
        return {"signal": "HOLD", "score": 0, "reasons": ["Insufficient data"]}

    # RSI signal
    rsi = calculate_rsi(df)
    last_rsi = rsi.iloc[-1]
    if last_rsi < 30:
        score += 2
        reasons.append(f"RSI oversold ({last_rsi:.1f}) → Bullish")
    elif last_rsi > 70:
        score -= 2
        reasons.append(f"RSI overbought ({last_rsi:.1f}) → Bearish")
    else:
        reasons.append(f"RSI neutral ({last_rsi:.1f})")

    # MACD signal
    macd_df = calculate_macd(df)
    if macd_df["MACD"].iloc[-1] > macd_df["Signal"].iloc[-1]:
        score += 1
        reasons.append("MACD above signal line → Bullish")
    else:
        score -= 1
        reasons.append("MACD below signal line → Bearish")

    # SMA crossover
    sma_20 = calculate_sma(df, 20)
    sma_50 = calculate_sma(df, 50)
    if sma_20.iloc[-1] > sma_50.iloc[-1]:
        score += 1
        reasons.append("SMA20 > SMA50 → Bullish")
    else:
        score -= 1
        reasons.append("SMA20 < SMA50 → Bearish")

    # Bollinger Bands
    bb = calculate_bollinger_bands(df)
    last_close = df["Close"].iloc[-1]
    if last_close < bb["BB_Lower"].iloc[-1]:
        score += 1
        reasons.append("Price below lower Bollinger Band → Potential reversal up")
    elif last_close > bb["BB_Upper"].iloc[-1]:
        score -= 1
        reasons.append("Price above upper Bollinger Band → Potential reversal down")

    # Stochastic
    stoch = calculate_stochastic(df)
    if stoch["%K"].iloc[-1] < 20 and stoch["%D"].iloc[-1] < 20:
        score += 1
        reasons.append("Stochastic oversold → Bullish")
    elif stoch["%K"].iloc[-1] > 80 and stoch["%D"].iloc[-1] > 80:
        score -= 1
        reasons.append("Stochastic overbought → Bearish")

    if score >= 3:
        signal = "STRONG BUY"
    elif score >= 1:
        signal = "BUY"
    elif score <= -3:
        signal = "STRONG SELL"
    elif score <= -1:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {"signal": signal, "score": score, "reasons": reasons}
