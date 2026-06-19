"""
News & Sentiment Analysis page.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.data_fetcher import get_stock_info, get_historical_data
from utils.sentiment import get_news_rss, get_news_api, get_sentiment_summary, build_sentiment_df
from utils.charts import sentiment_gauge, CHART_THEME


def render(ticker: str):
    st.markdown('<h1 class="main-header">📰 News & Sentiment Analysis</h1>', unsafe_allow_html=True)

    if not ticker:
        st.warning("Enter a ticker symbol in the sidebar.")
        return

    with st.spinner("Loading stock info..."):
        info = get_stock_info(ticker)

    company_name = info.get("longName") or info.get("shortName") or ticker
    st.markdown(f"### Sentiment for {company_name} ({ticker})")

    with st.spinner("Fetching news articles..."):
        # Try NewsAPI first, fall back to RSS
        articles = get_news_api(ticker, company_name)
        if not articles:
            articles = get_news_rss(ticker, company_name)

    if not articles:
        st.warning("No news articles found. Check your NEWS_API_KEY in .env or try a different ticker.")
        articles = []

    # ── Sentiment Analysis ─────────────────────────────────────────────────
    with st.spinner("Analyzing sentiment..."):
        summary = get_sentiment_summary(articles)

    # ── Summary Cards ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Sentiment Overview")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.plotly_chart(
            sentiment_gauge(summary["avg_score"]),
            use_container_width=True,
        )

    with col2:
        overall = summary["overall"]
        avg = summary["avg_score"]
        total = summary["total"]

        st.markdown(f"""
        <div style="background:#1e1e2e;border-radius:12px;padding:1.5rem;border:1px solid #333;">
            <h3 style="margin-top:0;">{overall}</h3>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:1rem;">
                <div style="text-align:center;">
                    <div style="color:#00c853;font-size:1.8rem;font-weight:700;">{summary['positive']}</div>
                    <div style="color:#aaa;">Positive</div>
                </div>
                <div style="text-align:center;">
                    <div style="color:#ffd600;font-size:1.8rem;font-weight:700;">{summary['neutral']}</div>
                    <div style="color:#aaa;">Neutral</div>
                </div>
                <div style="text-align:center;">
                    <div style="color:#ff1744;font-size:1.8rem;font-weight:700;">{summary['negative']}</div>
                    <div style="color:#aaa;">Negative</div>
                </div>
            </div>
            <div style="margin-top:1rem;color:#aaa;">Total articles analyzed: <strong style="color:white;">{total}</strong></div>
            <div style="color:#aaa;">Average sentiment score: <strong style="color:white;">{avg:.4f}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Sentiment Distribution Bar ─────────────────────────────────────────
    if total > 0:
        pos_pct = (summary["positive"] / total) * 100
        neu_pct = (summary["neutral"] / total) * 100
        neg_pct = (summary["negative"] / total) * 100

        fig_dist = go.Figure(data=[
            go.Bar(name="Positive 🟢", x=["Sentiment Distribution"], y=[pos_pct],
                   marker_color="#00c853", text=f"{pos_pct:.0f}%", textposition="auto"),
            go.Bar(name="Neutral 🟡", x=["Sentiment Distribution"], y=[neu_pct],
                   marker_color="#ffd600", text=f"{neu_pct:.0f}%", textposition="auto"),
            go.Bar(name="Negative 🔴", x=["Sentiment Distribution"], y=[neg_pct],
                   marker_color="#ff1744", text=f"{neg_pct:.0f}%", textposition="auto"),
        ])
        fig_dist.update_layout(
            barmode="stack", template=CHART_THEME, height=200,
            showlegend=True,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # ── Sentiment vs Price ─────────────────────────────────────────────────
    st.markdown("### 📈 Sentiment Context vs Price")
    df_price = get_historical_data(ticker, period="1mo")
    if not df_price.empty:
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=df_price.index, y=df_price["Close"],
            name="Price", mode="lines",
            line=dict(color="#00d4ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
        ))

        # Add sentiment annotation
        avg = summary["avg_score"]
        color = "#00c853" if avg >= 0.05 else ("#ff1744" if avg <= -0.05 else "#ffd600")
        fig_price.add_annotation(
            x=df_price.index[-1], y=df_price["Close"].iloc[-1],
            text=f"Sentiment: {summary['overall'].split()[0]}",
            showarrow=True, arrowhead=1,
            bgcolor=color, font=dict(color="white", size=11),
        )
        fig_price.update_layout(
            title=f"{ticker} — 1 Month Price with Current Sentiment",
            template=CHART_THEME, height=350,
        )
        st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("---")

    # ── News Articles with Sentiment ───────────────────────────────────────
    st.markdown("### 📰 Latest News Articles")

    if not summary["articles_with_sentiment"]:
        st.info("No articles to display.")
        return

    # Filter controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_sentiment = st.selectbox(
            "Filter by Sentiment",
            ["All", "Positive", "Negative", "Neutral"],
        )
    with col_f2:
        sort_by = st.selectbox("Sort by", ["Sentiment Score (High)", "Sentiment Score (Low)"])

    analyzed_articles = summary["articles_with_sentiment"]
    if filter_sentiment != "All":
        analyzed_articles = [
            a for a in analyzed_articles
            if filter_sentiment in a["sentiment"]["label"]
        ]

    reverse = sort_by == "Sentiment Score (High)"
    analyzed_articles = sorted(
        analyzed_articles,
        key=lambda x: x["sentiment"]["combined"],
        reverse=reverse,
    )

    # Display articles
    for article in analyzed_articles[:20]:
        sent = article["sentiment"]
        score = sent["combined"]
        label = sent["label"]
        color = "#00c853" if score >= 0.05 else ("#ff1744" if score <= -0.05 else "#ffd600")

        title = article.get("title", "No title")
        summary_text = article.get("summary", "")[:200]
        source = article.get("source", "Unknown")
        published = article.get("published", "")[:20] if article.get("published") else ""
        url = article.get("url", "#")

        with st.container():
            st.markdown(f"""
            <div style="background:#1e1e2e;border-radius:8px;padding:1rem;margin-bottom:0.5rem;border-left:4px solid {color};">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div style="flex:1;">
                        <a href="{url}" target="_blank" style="color:white;text-decoration:none;font-weight:600;">{title}</a>
                        <div style="color:#aaa;font-size:0.85rem;margin-top:0.3rem;">{summary_text}</div>
                        <div style="color:#666;font-size:0.75rem;margin-top:0.3rem;">{source} · {published}</div>
                    </div>
                    <div style="text-align:center;min-width:100px;margin-left:1rem;">
                        <div style="color:{color};font-weight:700;">{label.split()[0]}</div>
                        <div style="color:{color};font-size:0.85rem;">Score: {score:.3f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Sentiment DataFrame Export ─────────────────────────────────────────
    st.markdown("---")
    with st.expander("📊 Sentiment Data Table"):
        sent_df = build_sentiment_df(summary["articles_with_sentiment"])
        st.dataframe(sent_df[["Title", "Source", "Sentiment", "Score", "Published"]],
                     hide_index=True, use_container_width=True)
        csv = sent_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Sentiment Data",
            data=csv,
            file_name=f"{ticker}_sentiment.csv",
            mime="text/csv",
        )
