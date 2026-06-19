"""
Machine Learning models for stock price prediction.
Linear Regression, Random Forest, and LSTM.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def prepare_features(df: pd.DataFrame, feature_days: int = 30) -> pd.DataFrame:
    """Create feature-engineered dataframe for ML models."""
    data = df[["Close", "Volume", "High", "Low", "Open"]].copy()

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        data[f"Close_lag_{lag}"] = data["Close"].shift(lag)

    # Rolling statistics
    for window in [5, 10, 20]:
        data[f"SMA_{window}"] = data["Close"].rolling(window).mean()
        data[f"STD_{window}"] = data["Close"].rolling(window).std()

    # RSI-like momentum
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    data["RSI"] = 100 - (100 / (1 + rs))

    # Price change features
    data["Daily_Return"] = data["Close"].pct_change()
    data["Volatility"] = data["Daily_Return"].rolling(10).std()
    data["Price_Range"] = data["High"] - data["Low"]
    data["Price_Position"] = (data["Close"] - data["Low"]) / (data["High"] - data["Low"] + 1e-10)

    data.dropna(inplace=True)
    return data


def linear_regression_predict(df: pd.DataFrame, forecast_days: int = 30) -> dict:
    """Linear Regression price prediction."""
    data = prepare_features(df)
    if len(data) < 60:
        return {"error": "Not enough data for prediction (need at least 60 rows)"}

    feature_cols = [c for c in data.columns if c != "Close"]
    X = data[feature_cols].values
    y = data["Close"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Future prediction (naive: use last row features)
    last_features = X[-1].reshape(1, -1)
    future_prices = []
    current = last_features.copy()
    base_price = y[-1]

    for i in range(forecast_days):
        pred = model.predict(current)[0]
        future_prices.append(pred)
        # Shift lag features
        current = current.copy()
        current[0][0] = pred  # update Close_lag_1 roughly

    metrics = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "R2": round(r2_score(y_test, y_pred), 4),
        "MAPE": round(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100, 2),
    }

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="B")

    return {
        "model": "Linear Regression",
        "test_actual": y_test.tolist(),
        "test_predicted": y_pred.tolist(),
        "future_dates": future_dates.tolist(),
        "future_prices": future_prices,
        "metrics": metrics,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def random_forest_predict(df: pd.DataFrame, forecast_days: int = 30) -> dict:
    """Random Forest price prediction."""
    data = prepare_features(df)
    if len(data) < 60:
        return {"error": "Not enough data for prediction"}

    feature_cols = [c for c in data.columns if c != "Close"]
    X = data[feature_cols].values
    y = data["Close"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Feature importances
    importances = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]

    # Future prices
    future_prices = []
    current = X[-1].copy()
    for i in range(forecast_days):
        pred = model.predict(current.reshape(1, -1))[0]
        future_prices.append(pred)
        current[0] = pred

    metrics = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "R2": round(r2_score(y_test, y_pred), 4),
        "MAPE": round(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100, 2),
    }

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="B")

    return {
        "model": "Random Forest",
        "test_actual": y_test.tolist(),
        "test_predicted": y_pred.tolist(),
        "future_dates": future_dates.tolist(),
        "future_prices": future_prices,
        "metrics": metrics,
        "feature_importance": top_features,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def lstm_predict(df: pd.DataFrame, forecast_days: int = 30, lookback: int = 60) -> dict:
    """LSTM Deep Learning price prediction."""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        return {"error": "TensorFlow not installed. Run: pip install tensorflow"}

    if len(df) < lookback + 30:
        return {"error": f"Need at least {lookback + 30} days of data for LSTM"}

    close_prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback : i, 0])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0,
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Future prediction
    last_sequence = scaled[-lookback:].reshape(1, lookback, 1)
    future_prices = []
    current_seq = last_sequence.copy()

    for _ in range(forecast_days):
        pred_scaled = model.predict(current_seq, verbose=0)[0][0]
        future_prices.append(scaler.inverse_transform([[pred_scaled]])[0][0])
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, 0] = pred_scaled

    metrics = {
        "MAE": round(mean_absolute_error(y_actual, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_actual, y_pred)), 4),
        "R2": round(r2_score(y_actual, y_pred), 4),
        "MAPE": round(np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-10))) * 100, 2),
    }

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="B")

    return {
        "model": "LSTM (Deep Learning)",
        "test_actual": y_actual.tolist(),
        "test_predicted": y_pred.tolist(),
        "future_dates": future_dates.tolist(),
        "future_prices": [round(p, 2) for p in future_prices],
        "metrics": metrics,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def generate_recommendation(
    current_price: float,
    predicted_price: float,
    technical_score: int,
    pe_ratio: float = None,
) -> dict:
    """Generate Buy/Sell/Hold recommendation based on ML + technical analysis."""
    price_upside = ((predicted_price - current_price) / current_price) * 100

    rec_score = 0
    reasons = []

    # ML prediction signal
    if price_upside > 10:
        rec_score += 3
        reasons.append(f"ML predicts +{price_upside:.1f}% upside → Strong Buy signal")
    elif price_upside > 3:
        rec_score += 1
        reasons.append(f"ML predicts +{price_upside:.1f}% upside → Mild Buy signal")
    elif price_upside < -10:
        rec_score -= 3
        reasons.append(f"ML predicts {price_upside:.1f}% downside → Strong Sell signal")
    elif price_upside < -3:
        rec_score -= 1
        reasons.append(f"ML predicts {price_upside:.1f}% downside → Mild Sell signal")
    else:
        reasons.append(f"ML predicts {price_upside:.1f}% change → Neutral")

    # Technical score
    rec_score += technical_score
    if technical_score >= 2:
        reasons.append(f"Technical indicators bullish (score: {technical_score})")
    elif technical_score <= -2:
        reasons.append(f"Technical indicators bearish (score: {technical_score})")

    # Valuation (P/E if available)
    if pe_ratio is not None:
        if 0 < pe_ratio < 15:
            rec_score += 1
            reasons.append(f"P/E ratio {pe_ratio:.1f} — potentially undervalued")
        elif pe_ratio > 40:
            rec_score -= 1
            reasons.append(f"P/E ratio {pe_ratio:.1f} — potentially overvalued")

    # Final recommendation
    if rec_score >= 4:
        recommendation = "STRONG BUY"
        color = "🟢"
        confidence = min(95, 60 + rec_score * 5)
    elif rec_score >= 2:
        recommendation = "BUY"
        color = "🟩"
        confidence = min(80, 55 + rec_score * 4)
    elif rec_score <= -4:
        recommendation = "STRONG SELL"
        color = "🔴"
        confidence = min(95, 60 + abs(rec_score) * 5)
    elif rec_score <= -2:
        recommendation = "SELL"
        color = "🟥"
        confidence = min(80, 55 + abs(rec_score) * 4)
    else:
        recommendation = "HOLD"
        color = "🟡"
        confidence = 50

    return {
        "recommendation": recommendation,
        "color": color,
        "confidence": confidence,
        "score": rec_score,
        "price_upside": round(price_upside, 2),
        "reasons": reasons,
    }
