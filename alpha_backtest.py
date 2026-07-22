# -------------------------------------------------------------
# Optimal Portfolio Backtest Tool — Compact UI Version
# -------------------------------------------------------------

import streamlit as st
from pandas_datareader import data as pdr
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, date, timedelta
import plotly.express as px
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from zoneinfo import ZoneInfo

Eastern_time = datetime.now(ZoneInfo("America/New_York"))
AUTO_REFRESH_SECONDS = 18000  # 5 hours

# -------------------------------------------------------------
# PAGE CONFIG  (must be first Streamlit call)
# -------------------------------------------------------------
st.set_page_config(page_title="Alpha Backtest", layout="wide")

# =============================================================================
# DESIGN SYSTEM
# -----------------------------------------------------------------------------
# Same research-desk aesthetic used across the Risk Management Model and
# Alpha Dashboard: ink slate + ledger ivory, deep emerald / antique gold
# accents, Fraunces for display type, Inter for body text, IBM Plex Mono
# for figures. Kept quiet (low contrast, compact spacing) to match.
# =============================================================================

PALETTE = {
    "ink": "#2B3B50",        # soft slate navy — sidebar, headings
    "ink_2": "#374B65",      # secondary ink surface
    "paper": "#F6F4EE",      # warm ivory — page background
    "paper_2": "#EFEBE0",    # card / metric surface
    "rule": "rgba(43,59,80,0.10)",   # hairline dividers
    "text": "#33404F",       # body text on paper
    "muted": "#697787",      # secondary text
    "paper_text": "#D9D4C7", # text on ink surfaces
    "emerald": "#33604F",    # primary accent — gains, confidence
    "emerald_soft": "rgba(51,96,79,0.08)",
    "gold": "#B0925F",       # secondary accent — highlights, rules
    "gold_soft": "rgba(176,146,95,0.12)",
    "burgundy": "#8A4A4A",   # risk / loss accent
    "burgundy_soft": "rgba(138,74,74,0.08)",
}

PLOTLY_COLORWAY = [
    PALETTE["emerald"], PALETTE["gold"], PALETTE["burgundy"],
    "#3F6B57", "#8C6E4A", "#4A5A73",
]


def inject_design_system():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        .stApp {{
            background: {PALETTE["paper"]};
            color: {PALETTE["text"]};
            font-size: 0.92rem;
        }}

        .block-container {{
            padding-top: 1.4rem !important;
            padding-bottom: 1.3rem !important;
            max-width: 1300px;
        }}

        /* ---------- Typography ---------- */
        h1, h2, h3, h4 {{
            font-family: 'Fraunces', serif !important;
            color: {PALETTE["ink"]} !important;
            font-weight: 500 !important;
            letter-spacing: -0.01em;
            margin-top: 0.2rem;
            margin-bottom: 0.2rem;
        }}
        h4 {{
            border-bottom: 1px solid {PALETTE["rule"]};
            padding-bottom: 0.25rem;
            margin-top: 1rem !important;
            font-size: 1.05rem !important;
        }}
        .eyebrow {{
            display: block;
            font-family: 'Inter', sans-serif;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            color: {PALETTE["gold"]};
            margin-bottom: 0.1rem;
        }}

        /* ---------- Masthead ---------- */
        .masthead {{
            border-top: 2px solid {PALETTE["ink"]};
            border-bottom: 1px solid {PALETTE["rule"]};
            padding: 0.5rem 0 0.6rem 0;
            margin-bottom: 0.9rem;
            text-align: center;
        }}
        .masthead .eyebrow {{ margin-bottom: 0.2rem; text-align: center; }}
        .masthead h1 {{
            font-size: 1.6rem !important;
            margin: 0 !important;
            line-height: 1.15;
        }}
        .masthead .dek {{
            font-family: 'Inter', sans-serif;
            color: {PALETTE["muted"]};
            font-size: 0.82rem;
            margin-top: 0.2rem;
        }}

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {{
            background: {PALETTE["ink"]};
        }}
        [data-testid="stSidebar"] * {{
            color: {PALETTE["paper_text"]} !important;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            font-family: 'Fraunces', serif !important;
            font-weight: 500 !important;
            color: {PALETTE["paper_text"]} !important;
            border-bottom: 1px solid rgba(217,212,199,0.14);
            padding-bottom: 0.3rem;
            font-size: 1.05rem !important;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: rgba(217,212,199,0.12) !important;
            margin: 0.6rem 0 !important;
        }}
        [data-testid="stSidebar"] label {{ color: {PALETTE["paper_text"]} !important; opacity: 0.8; }}

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: {PALETTE["ink_2"]} !important;
            border: 1px solid rgba(217,212,199,0.16) !important;
            color: {PALETTE["paper_text"]} !important;
            border-radius: 4px !important;
        }}

        /* ---------- Buttons ---------- */
        .stButton > button, button[kind="primary"] {{
            background: {PALETTE["emerald"]} !important;
            color: {PALETTE["paper_text"]} !important;
            border: 1px solid {PALETTE["emerald"]} !important;
            border-radius: 4px !important;
            font-weight: 500 !important;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            font-size: 0.74rem !important;
            padding: 0.35rem 0.85rem !important;
        }}
        .stButton > button:hover, button[kind="primary"]:hover {{
            background: {PALETTE["ink"]} !important;
            border-color: {PALETTE["gold"]} !important;
            color: {PALETTE["gold"]} !important;
        }}

        /* ---------- Metrics ---------- */
        [data-testid="stMetric"] {{
            background: {PALETTE["paper_2"]};
            border: 1px solid {PALETTE["rule"]};
            border-radius: 6px;
            padding: 0.5rem 0.65rem 0.4rem 0.65rem !important;
        }}
        [data-testid="stMetricLabel"] {{
            font-family: 'Inter', sans-serif !important;
            font-size: 0.66rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {PALETTE["muted"]} !important;
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'IBM Plex Mono', monospace !important;
            color: {PALETTE["ink"]} !important;
            font-weight: 500 !important;
            font-size: 1.25rem !important;
        }}

        /* ---------- Dataframes ---------- */
        [data-testid="stDataFrame"] {{
            border: 1px solid {PALETTE["rule"]};
            border-radius: 6px;
            overflow: hidden;
        }}

        /* ---------- Rules ---------- */
        hr {{ border-color: {PALETTE["rule"]} !important; margin: 0.6rem 0 !important; }}

        /* ---------- Section header block ---------- */
        .section-head h4 {{ margin-top: 0.4rem !important; }}

        /* ---------- General compaction ---------- */
        div[data-testid="stVerticalBlock"] {{ gap: 0.4rem; }}
        div[data-testid="stHorizontalBlock"] {{ gap: 0.6rem; }}
        .element-container {{ margin-bottom: 0.1rem !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def masthead(eyebrow, title, dek):
    st.markdown(
        f"""
        <div class="masthead">
            <span class="eyebrow">{eyebrow}</span>
            <h1>{title}</h1>
            <div class="dek">{dek}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(eyebrow, title):
    st.markdown(
        f"""
        <div class="section-head">
            <span class="eyebrow">{eyebrow}</span>
            <h4>{title}</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )


def themed_layout(fig, height=None, title=None):
    """Apply the house plotly theme to a figure in place, and return it."""
    layout_kwargs = dict(
        paper_bgcolor=PALETTE["paper"],
        plot_bgcolor=PALETTE["paper"],
        font=dict(family="Inter, sans-serif", color=PALETTE["text"], size=12),
        colorway=PLOTLY_COLORWAY,
        margin=dict(t=46 if title else 20, l=10, r=10, b=10),
        legend=dict(font=dict(family="Inter, sans-serif", size=11)),
        title_x=0.5,
    )
    if title:
        layout_kwargs["title"] = dict(
            text=title, x=0.5, font=dict(family="Fraunces, serif", size=15, color=PALETTE["ink"])
        )
    if height:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor="rgba(43,59,80,0.08)", zerolinecolor="rgba(43,59,80,0.15)", linecolor=PALETTE["rule"])
    fig.update_yaxes(gridcolor="rgba(43,59,80,0.08)", zerolinecolor="rgba(43,59,80,0.15)", linecolor=PALETTE["rule"])
    return fig


inject_design_system()

masthead(
    eyebrow="Alpha Desk &middot; Portfolio Optimization",
    title="Optimal Portfolio Backtest Tool",
    dek="Calculate optimal allocations, expected returns, and performance metrics",
)

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
st.sidebar.header("Configure Portfolio")

DEFAULT_PORTFOLIO = "^GSPC, SYM, CRWV, KRYS, AAPL, AMZN, BBAI, BEAM, CELC, COGT, CRSP, DYN, GOOG, INTC, NVDA, META, PLTR, TSLA"

def get_input():
    default_start_date = date(date.today().year - 1, 12, 31)
    default_end_date = date.today()

    start_date = st.sidebar.date_input("Start Date", default_start_date)
    end_date = st.sidebar.date_input("End Date", default_end_date)

    portfolio = st.sidebar.text_input(
        "Portfolio (format: NVDA, PLTR, GOOG)",
        DEFAULT_PORTFOLIO
    ).upper()

    total_funds = st.sidebar.number_input(
        "Total Funds to Invest",
        value=100000,
        step=5000,
        format='%d'
    )

    return start_date, end_date, portfolio, total_funds

@st.cache_data(ttl=300)
def fetch_data(symbols, start_date, end_date):
    end_date_adj = end_date + timedelta(days=1)
    try:
        df = yf.download(symbols, start=start_date, end=end_date_adj, progress=False)['Close']
        if len(symbols) == 1:
            df = pd.DataFrame(df)
            df.columns = [symbols[0]]
    except Exception as e:
        st.warning(f"Error fetching data: {e}")
        return pd.DataFrame()
    return df

def calculate_weights(symbols):
    return np.ones(len(symbols)) / len(symbols)

def plot_prices(df, title):
    fig = px.line(df)
    themed_layout(fig, height=380, title=title)
    fig.update_layout(yaxis_title="", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# USER INPUT
# -------------------------------------------------------------
start_date, end_date, portfolio, funds_to_invest = get_input()
symbols = [s.strip() for s in portfolio.split(',')]
funds_to_invest = max(funds_to_invest, 1)

if not symbols:
    st.error("Please enter at least one stock symbol.")
    st.stop()

df = fetch_data(symbols, start_date, end_date)

if df.empty or df.isnull().all().all():
    st.error("No valid stock data available. Please check symbols or date range.")
    st.stop()

df = df.ffill().bfill()
df_normalized = df / df.iloc[0]

# -------------------------------------------------------------
# PRICE CHART
# -------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
section_header("Trajectory", "Price Performance")

plot_prices(df_normalized, "Relative Daily Change Prices")

# -------------------------------------------------------------
# REFRESH CONTROLS
# -------------------------------------------------------------
if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0

manual_refresh = st.sidebar.button("Refresh")
auto_refresh = st.sidebar.checkbox("Auto Refresh (Every 5 Hours)", value=False)

if manual_refresh:
    st.session_state.refresh_counter += 1
    st.rerun()

if auto_refresh:
    st.markdown(
        f"""
        <script>
            setTimeout(function(){{
                window.location.reload();
            }}, {AUTO_REFRESH_SECONDS * 1000});
        </script>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------------
# PORTFOLIO ANALYSIS
# -------------------------------------------------------------
returns = df.pct_change().dropna()

if returns.empty:
    st.error("Insufficient data to calculate returns.")
    st.stop()

cov_matrix_annual = returns.cov() * 252
weights = calculate_weights(symbols)

portfolio_return = np.sum(returns.mean() * weights) * 252
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
portfolio_volatility = np.sqrt(portfolio_variance)

st.markdown("<hr>", unsafe_allow_html=True)
section_header("Metrics", "Portfolio Statistics")

col1, col2, col3 = st.columns(3)
col1.metric("Expected Return", f"{portfolio_return:.1%}")
col2.metric("Volatility", f"{portfolio_volatility:.1%}")
col3.metric("Variance", f"{portfolio_variance:.1%}")

# -------------------------------------------------------------
# OPTIMIZATION
# -------------------------------------------------------------
try:
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=funds_to_invest)
    allocation, leftover = da.greedy_portfolio()

    df_weights = pd.DataFrame(list(cleaned_weights.items()), columns=['Stocks', 'Weight'])
    df_allocation = pd.DataFrame(list(allocation.items()), columns=['Stocks', '#Shares'])
    merged_df = df_weights.merge(df_allocation, on='Stocks', how='left')
    merged_df['Weight'] *= 100

except Exception as e:
    st.error(f"Portfolio optimization failed: {e}")
    st.stop()

# -------------------------------------------------------------
# CALCULATIONS
# -------------------------------------------------------------
start_prices = df.iloc[0]
end_prices = df.iloc[-1]

price_df = pd.DataFrame({
    'Start Price': start_prices,
    'End Price': end_prices
})

merged_df.set_index('Stocks', inplace=True)
merged_df = merged_df.merge(price_df, left_index=True, right_index=True, how='left')

merged_df['Investment'] = (merged_df['Weight'] / 100) * funds_to_invest
merged_df['#Shares'] = np.floor(merged_df['Investment'] / merged_df['Start Price'])
merged_df['Investment'] = merged_df['#Shares'] * merged_df['Start Price']
merged_df['%Change'] = (merged_df['End Price'] / merged_df['Start Price'] - 1) * 100
merged_df['$Gain/Loss'] = merged_df['#Shares'] * (merged_df['End Price'] - merged_df['Start Price'])
merged_df['Total'] = merged_df['Investment'] + merged_df['$Gain/Loss']

total_gain = merged_df['$Gain/Loss'].sum()
total_invested = merged_df['Investment'].sum()
leftover = funds_to_invest - total_invested
final_portfolio_value = total_invested + total_gain + leftover
total_percent_gain = total_gain / total_invested if total_invested > 0 else 0

# -------------------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
section_header("Allocation", "Optimal Portfolio Allocation")

merged_df = merged_df.sort_values(by="Weight", ascending=False)

display_df = merged_df[['Weight', '#Shares', 'Start Price', 'End Price',
                        'Investment', '%Change', '$Gain/Loss', 'Total']]

from matplotlib.colors import LinearSegmentedColormap

RISK_CMAP = LinearSegmentedColormap.from_list(
    "risk_desk", [PALETTE["burgundy"], PALETTE["paper"], PALETTE["emerald"]]
)

styled_df = display_df.style.format({
    "Weight": "{:.1f}%",
    "#Shares": "{:.0f}",
    "Start Price": "${:,.2f}",
    "End Price": "${:,.2f}",
    "Investment": "${:,.0f}",
    "%Change": "{:.1f}%",
    "Total": "${:,.0f}",
    "$Gain/Loss": "${:,.0f}"
}).background_gradient(
    subset=["%Change", "$Gain/Loss"], cmap=RISK_CMAP
).set_properties(**{
    'text-align': 'center',
    'font-size': '12px'
})

st.dataframe(styled_df, use_container_width=True, height=350)

# -------------------------------------------------------------
# SUMMARY METRICS
# -------------------------------------------------------------
section_header("Summary", "Portfolio Summary")

colA, colB, colC, colD = st.columns(4)
colA.metric("Invested", f"${total_invested:,.0f}")
colB.metric("Uninvested", f"${leftover:,.0f}")
colC.metric("Gain / Loss", f"${total_gain:,.0f}", f"{total_percent_gain:.1%}")
colD.metric("Final Value", f"${final_portfolio_value:,.0f}")

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.sidebar.caption("ⓒ Franklin Chidi (FC) - MIT License")
st.sidebar.caption(f"Refreshed: {Eastern_time.strftime('%Y-%m-%d at %-I:%M %p (Eastern)')}")
