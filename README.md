# 📈 Stock Analysis AI

A comprehensive AI-powered stock analysis platform built with Python and Streamlit.

## Features

| Module | Features |
|---|---|
| 🏠 **Dashboard** | Real-time price, company profile, market cap, sector info, trading signals |
| 📊 **Historical Analysis** | OHLCV charts, custom date range, return statistics, volume analysis |
| 🔬 **Technical Analysis** | SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR, OBV, Support/Resistance |
| 💼 **Fundamental Analysis** | P/E, P/B, EPS, Margins, Debt/Equity, DCF valuation, income statements |
| ⚖️ **Stock Comparison** | Multi-stock performance, correlation matrix, risk vs return scatter |
| 🤖 **AI Predictions** | Linear Regression, Random Forest, LSTM forecasting with Buy/Sell/Hold recommendation |
| 📰 **News & Sentiment** | RSS/NewsAPI news feed, VADER + TextBlob sentiment analysis, sentiment gauge |

## Quick Start

### 1. Install dependencies
```bash
install.bat
# or manually:
pip install -r requirements.txt
```

### 2. (Optional) Add API keys
Copy `.env.example` to `.env` and add:
- `NEWS_API_KEY` — from [newsapi.org](https://newsapi.org) (free tier works)
- `ALPHA_VANTAGE_API_KEY` — from [alphavantage.co](https://alphavantage.co) (optional)

News works without keys via RSS feeds. API keys just improve coverage.

### 3. Run the app
```bash
run.bat
# or manually:
streamlit run app.py
```

The app opens at **http://localhost:8501**

## Project Structure

```
stock-analysis/
├── app.py                    # Main Streamlit app + routing
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
├── install.bat               # One-click installer
├── run.bat                   # One-click launcher
├── pages/
│   ├── dashboard.py          # Real-time dashboard
│   ├── historical.py         # Historical data & charts
│   ├── technical.py          # Technical indicators
│   ├── fundamental.py        # Financial ratios & statements
│   ├── comparison.py         # Multi-stock comparison
│   ├── predictions.py        # ML prediction models
│   └── sentiment.py          # News & sentiment analysis
└── utils/
    ├── data_fetcher.py        # yfinance data layer (cached)
    ├── technical_indicators.py # All indicator calculations
    ├── ml_models.py           # LR, RF, LSTM models
    ├── sentiment.py           # News fetching & NLP
    └── charts.py              # Reusable Plotly chart builders
```

## ML Models

- **Linear Regression** — Feature-engineered regression using lag prices, rolling stats, and RSI
- **Random Forest** — Ensemble model with 100 trees; shows feature importance
- **LSTM** — 2-layer deep learning model for sequence prediction (requires TensorFlow)

All models output:
- Test set predictions vs actual (backtest)
- Future price forecast with confidence band
- MAE, RMSE, R², MAPE accuracy metrics
- Buy / Sell / Hold recommendation with confidence score

## Notes

- All data is fetched from **Yahoo Finance** (free, no key needed)
- LSTM model requires `tensorflow` — if install fails on Python 3.8, try `pip install tensorflow==2.13.0`
- Data is cached (5-min for live prices, 1-hour for historical) to avoid rate limits
- Predictions are for **educational purposes only**

## Requirements

- Python 3.8+
- Internet connection (for live data)
- ~500MB disk space (for TensorFlow)
"# stock-analysis" 
