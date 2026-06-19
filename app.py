"""
Stock Analysis AI - Main Application Entry Point
Run with: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Stock Analysis AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #333;
    }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("## 📈 Stock Analysis AI")
    st.markdown("---")
    page = st.selectbox(
        "Navigate to",
        [
            "🏠 Dashboard",
            "📊 Historical Analysis",
            "🔬 Technical Analysis",
            "💼 Fundamental Analysis",
            "⚖️ Stock Comparison",
            "🤖 AI Predictions",
            "📰 News & Sentiment",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### Quick Search")
    ticker_input = st.text_input("Enter Ticker Symbol", value="AAPL", placeholder="e.g. AAPL, TSLA").upper().strip()
    st.markdown("---")
    st.caption("Data powered by Yahoo Finance")

# Route to pages
if page == "🏠 Dashboard":
    from pages.dashboard import render
    render(ticker_input)
elif page == "📊 Historical Analysis":
    from pages.historical import render
    render(ticker_input)
elif page == "🔬 Technical Analysis":
    from pages.technical import render
    render(ticker_input)
elif page == "💼 Fundamental Analysis":
    from pages.fundamental import render
    render(ticker_input)
elif page == "⚖️ Stock Comparison":
    from pages.comparison import render
    render()
elif page == "🤖 AI Predictions":
    from pages.predictions import render
    render(ticker_input)
elif page == "📰 News & Sentiment":
    from pages.sentiment import render
    render(ticker_input)
