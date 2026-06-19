"""
News fetching and sentiment analysis for stocks.
"""

import os
import feedparser
import requests
import streamlit as st
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd


analyzer = SentimentIntensityAnalyzer()


def get_news_rss(ticker: str, company_name: str = None) -> list:
    """Fetch news from multiple free RSS sources."""
    query = company_name or ticker
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en",
        f"https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    ]

    articles = []
    seen_titles = set()

    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:15]:
                title = entry.get("title", "")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    published = entry.get("published", "")
                    summary = entry.get("summary", entry.get("description", ""))
                    link = entry.get("link", "")
                    articles.append({
                        "title": title,
                        "summary": summary[:500] if summary else "",
                        "url": link,
                        "published": published,
                        "source": feed.feed.get("title", "Unknown"),
                    })
        except Exception:
            continue

    return articles[:30]


def get_news_api(ticker: str, company_name: str = None) -> list:
    """Fetch news using NewsAPI if key is available."""
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key or api_key == "your_news_api_key_here":
        return []

    query = company_name or ticker
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "pageSize": 20,
        "language": "en",
        "apiKey": api_key,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            articles = []
            for a in data.get("articles", []):
                articles.append({
                    "title": a.get("title", ""),
                    "summary": a.get("description", ""),
                    "url": a.get("url", ""),
                    "published": a.get("publishedAt", ""),
                    "source": a.get("source", {}).get("name", "Unknown"),
                })
            return articles
    except Exception:
        pass
    return []


def analyze_sentiment(text: str) -> dict:
    """Run VADER + TextBlob sentiment analysis on text."""
    if not text:
        return {"compound": 0, "label": "Neutral", "vader": {}, "textblob": 0}

    # VADER
    vader_scores = analyzer.polarity_scores(text)
    compound = vader_scores["compound"]

    # TextBlob
    tb = TextBlob(text)
    tb_polarity = tb.sentiment.polarity

    # Combined score
    combined = (compound + tb_polarity) / 2

    if combined >= 0.05:
        label = "Positive 🟢"
    elif combined <= -0.05:
        label = "Negative 🔴"
    else:
        label = "Neutral 🟡"

    return {
        "compound": round(compound, 4),
        "textblob": round(tb_polarity, 4),
        "combined": round(combined, 4),
        "label": label,
        "vader": vader_scores,
    }


def get_sentiment_summary(articles: list) -> dict:
    """Analyze all articles and return aggregate sentiment."""
    if not articles:
        return {
            "overall": "No news found",
            "avg_score": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "articles_with_sentiment": [],
        }

    analyzed = []
    scores = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for article in articles:
        text = f"{article.get('title', '')} {article.get('summary', '')}"
        sentiment = analyze_sentiment(text)
        article_data = {**article, "sentiment": sentiment}
        analyzed.append(article_data)
        scores.append(sentiment["combined"])

        if sentiment["combined"] >= 0.05:
            positive_count += 1
        elif sentiment["combined"] <= -0.05:
            negative_count += 1
        else:
            neutral_count += 1

    avg_score = sum(scores) / len(scores) if scores else 0

    if avg_score >= 0.1:
        overall = "Bullish Sentiment 🟢"
    elif avg_score >= 0.02:
        overall = "Slightly Bullish 🟩"
    elif avg_score <= -0.1:
        overall = "Bearish Sentiment 🔴"
    elif avg_score <= -0.02:
        overall = "Slightly Bearish 🟥"
    else:
        overall = "Neutral Sentiment 🟡"

    return {
        "overall": overall,
        "avg_score": round(avg_score, 4),
        "positive": positive_count,
        "negative": negative_count,
        "neutral": neutral_count,
        "total": len(analyzed),
        "articles_with_sentiment": analyzed,
    }


def build_sentiment_df(articles_with_sentiment: list) -> pd.DataFrame:
    """Convert analyzed articles to a DataFrame."""
    rows = []
    for a in articles_with_sentiment:
        rows.append({
            "Title": a.get("title", ""),
            "Source": a.get("source", ""),
            "Published": a.get("published", ""),
            "Sentiment": a["sentiment"]["label"],
            "Score": a["sentiment"]["combined"],
            "URL": a.get("url", ""),
        })
    return pd.DataFrame(rows)
