"""
Fundamental Analysis page — financial ratios, earnings, revenue, margins.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.data_fetcher import get_stock_info, get_financials
from utils.charts import CHART_THEME


def render(ticker: str):
    st.markdown('<h1 class="main-header">💼 Fundamental Analysis</h1>', unsafe_allow_html=True)

    if not ticker:
        st.warning("Enter a ticker symbol in the sidebar.")
        return

    with st.spinner("Loading fundamental data..."):
        info = get_stock_info(ticker)
        financials = get_financials(ticker)

    if not info:
        st.error(f"Could not load data for '{ticker}'.")
        return

    company_name = info.get("longName") or ticker
    st.markdown(f"### {company_name} ({ticker})")

    # ── Valuation Ratios ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Key Valuation Metrics")

    ratios = [
        ("P/E Ratio (TTM)", info.get("trailingPE"), "x", "15-25 is typical for S&P 500"),
        ("Forward P/E", info.get("forwardPE"), "x", "Analysts' expected earnings"),
        ("P/B Ratio", info.get("priceToBook"), "x", "<1 may indicate undervalue"),
        ("P/S Ratio", info.get("priceToSalesTrailing12Months"), "x", "Lower = cheaper relative to revenue"),
        ("EV/EBITDA", info.get("enterpriseToEbitda"), "x", "Lower = potentially undervalued"),
        ("EV/Revenue", info.get("enterpriseToRevenue"), "x", ""),
    ]

    cols = st.columns(3)
    for i, (name, val, unit, note) in enumerate(ratios):
        with cols[i % 3]:
            display = f"{val:.2f}{unit}" if val is not None else "N/A"
            st.metric(name, display, help=note)

    st.markdown("---")

    # ── Per-Share Metrics ──────────────────────────────────────────────────
    st.markdown("### 💰 Per-Share Metrics")
    cols2 = st.columns(4)
    per_share = [
        ("EPS (TTM)", info.get("trailingEps"), "$"),
        ("EPS (Forward)", info.get("forwardEps"), "$"),
        ("Book Value/Share", info.get("bookValue"), "$"),
        ("Revenue/Share", info.get("revenuePerShare"), "$"),
    ]
    for i, (name, val, prefix) in enumerate(per_share):
        display = f"{prefix}{val:.2f}" if val is not None else "N/A"
        cols2[i].metric(name, display)

    st.markdown("---")

    # ── Profitability & Margins ────────────────────────────────────────────
    st.markdown("### 📈 Profitability & Margins")
    margin_cols = st.columns(4)
    margins = [
        ("Gross Margin", info.get("grossMargins")),
        ("Operating Margin", info.get("operatingMargins")),
        ("Net Profit Margin", info.get("profitMargins")),
        ("Return on Equity", info.get("returnOnEquity")),
    ]
    for i, (name, val) in enumerate(margins):
        display = f"{val * 100:.2f}%" if val is not None else "N/A"
        margin_cols[i].metric(name, display)

    # Margin bar chart
    margin_vals = [(name, val * 100) for name, val in margins if val is not None]
    if margin_vals:
        names, values = zip(*margin_vals)
        colors = ["#00c853" if v > 0 else "#ff1744" for v in values]
        fig_margins = go.Figure(go.Bar(
            x=list(names), y=list(values),
            marker_color=colors, text=[f"{v:.1f}%" for v in values],
            textposition="auto",
        ))
        fig_margins.update_layout(
            title="Margin Analysis", template=CHART_THEME, height=350,
            yaxis_title="Percentage (%)",
        )
        st.plotly_chart(fig_margins, use_container_width=True)

    st.markdown("---")

    # ── Debt & Liquidity ───────────────────────────────────────────────────
    st.markdown("### 🏦 Debt & Liquidity")
    debt_cols = st.columns(4)
    debt_metrics = [
        ("Debt/Equity Ratio", info.get("debtToEquity"), "", "Lower is safer"),
        ("Current Ratio", info.get("currentRatio"), "x", ">1 means solvent short-term"),
        ("Quick Ratio", info.get("quickRatio"), "x", ">1 is healthy"),
        ("Total Cash", info.get("totalCash"), "$", ""),
    ]
    for i, (name, val, suffix, note) in enumerate(debt_metrics):
        if val is not None:
            if suffix == "$":
                if val >= 1e9:
                    display = f"${val/1e9:.2f}B"
                elif val >= 1e6:
                    display = f"${val/1e6:.2f}M"
                else:
                    display = f"${val:,.0f}"
            else:
                display = f"{val:.2f}{suffix}"
        else:
            display = "N/A"
        debt_cols[i].metric(name, display, help=note)

    st.markdown("---")

    # ── Dividends & Growth ─────────────────────────────────────────────────
    st.markdown("### 💵 Dividends & Growth")
    div_cols = st.columns(4)
    div_metrics = [
        ("Dividend Yield", info.get("dividendYield"), "%"),
        ("Dividend Rate", info.get("dividendRate"), "$"),
        ("5Y Revenue CAGR", info.get("revenueGrowth"), "%"),
        ("Earnings Growth", info.get("earningsGrowth"), "%"),
    ]
    for i, (name, val, unit) in enumerate(div_metrics):
        if val is not None:
            display = f"{val * 100:.2f}{unit}" if unit == "%" else f"${val:.2f}"
        else:
            display = "N/A"
        div_cols[i].metric(name, display)

    st.markdown("---")

    # ── Income Statement ───────────────────────────────────────────────────
    income = financials.get("income_stmt")
    if income is not None and not income.empty:
        st.markdown("### 📋 Income Statement (Annual)")
        try:
            display_income = income.copy()
            display_income = display_income / 1e6  # millions
            display_income = display_income.round(2)

            key_rows = [r for r in ["Total Revenue", "Gross Profit", "Operating Income",
                                    "Net Income", "EBITDA"] if r in display_income.index]
            if key_rows:
                subset = display_income.loc[key_rows]
                subset.columns = [str(c)[:10] for c in subset.columns]

                fig_income = go.Figure()
                colors = ["#00d4ff", "#7b2ff7", "#00c853", "#ff9800", "#e91e63"]
                for i, row in enumerate(subset.index):
                    fig_income.add_trace(go.Bar(
                        name=row, x=subset.columns.tolist(),
                        y=subset.loc[row].tolist(),
                        marker_color=colors[i % len(colors)],
                    ))
                fig_income.update_layout(
                    title="Annual Financials (Millions USD)", barmode="group",
                    template=CHART_THEME, height=400,
                )
                st.plotly_chart(fig_income, use_container_width=True)

                # Table
                with st.expander("View Raw Data"):
                    st.dataframe(subset, use_container_width=True)
        except Exception as e:
            st.caption(f"Could not render income statement: {e}")

    # ── Balance Sheet ──────────────────────────────────────────────────────
    balance = financials.get("balance_sheet")
    if balance is not None and not balance.empty:
        with st.expander("🏗️ Balance Sheet (Annual)"):
            try:
                display_bs = balance.copy() / 1e9  # billions
                display_bs = display_bs.round(3)
                display_bs.columns = [str(c)[:10] for c in display_bs.columns]
                st.dataframe(display_bs, use_container_width=True)
            except Exception:
                st.write("Balance sheet data unavailable.")

    # ── DCF Valuation (simplified) ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔮 Simplified DCF Valuation")
    with st.expander("Run DCF Estimate"):
        fcf = info.get("freeCashflow")
        shares = info.get("sharesOutstanding")
        if fcf and shares:
            col_dcf1, col_dcf2, col_dcf3 = st.columns(3)
            growth_rate = col_dcf1.slider("Growth Rate (yr 1-5)", 0.0, 0.30, 0.10, 0.01)
            terminal_rate = col_dcf2.slider("Terminal Growth Rate", 0.01, 0.05, 0.03, 0.005)
            discount_rate = col_dcf3.slider("Discount Rate (WACC)", 0.05, 0.20, 0.10, 0.005)

            cash_flows = []
            cf = fcf
            for year in range(1, 6):
                cf *= (1 + growth_rate)
                discounted = cf / ((1 + discount_rate) ** year)
                cash_flows.append(discounted)

            terminal_value = (cash_flows[-1] * (1 + terminal_rate)) / (discount_rate - terminal_rate)
            terminal_pv = terminal_value / ((1 + discount_rate) ** 5)
            total_pv = sum(cash_flows) + terminal_pv

            dcf_per_share = total_pv / shares
            current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            margin_of_safety = ((dcf_per_share - current_price) / dcf_per_share) * 100 if dcf_per_share else 0

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("DCF Intrinsic Value", f"${dcf_per_share:.2f}/share")
            col_r2.metric("Current Price", f"${current_price:.2f}/share")
            col_r3.metric("Margin of Safety", f"{margin_of_safety:.1f}%",
                          delta="Undervalued" if margin_of_safety > 0 else "Overvalued")
        else:
            st.info("Free cash flow data not available for DCF calculation.")
