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

# -------------------------------------------------------------
# GLOBAL COMPACT CSS
# -------------------------------------------------------------
st.markdown("""
    <style>
        .block-container {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        h1, h2, h3, h4 {
            margin-top: 0.2rem;
            margin-bottom: 0.2rem;
        }
        .stMetric {
            padding: 0 !important;
        }
        hr {
            margin: 6px 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# COMPACT HEADER
# -------------------------------------------------------------
st.markdown("""
    <div style="text-align:center; padding: 10px 0 8px 0;">
        <h2 style="color:#4A4A4A; margin-bottom:2px; font-size:36px;">
            Optimal Portfolio Backtest Tool
        </h2>
        <p style="color:#777; font-size:16px; margin-top:0;">
            Calculate optimal allocations, expected returns, and performance metrics
        </p>
    </div>
""", unsafe_allow_html=True)

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
    fig.update_layout(
        title=title,
        yaxis_title="",
        xaxis_title="",
        title_x=0.5,
        font=dict(family="Arial", size=11),
        height=450  # compact chart height
    )
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
st.markdown("<h4 style='margin-bottom:4px;'>📈 Price Performance</h4>", unsafe_allow_html=True)

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
st.markdown("<h4 style='margin-bottom:4px;'>📊 Portfolio Statistics</h4>", unsafe_allow_html=True)

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
st.markdown("<h4 style='margin-bottom:4px;'>📌 Optimal Portfolio Allocation</h4>", unsafe_allow_html=True)

merged_df = merged_df.sort_values(by="Weight", ascending=False)

display_df = merged_df[['Weight', '#Shares', 'Start Price', 'End Price',
                        'Investment', '%Change', '$Gain/Loss', 'Total']]

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
    subset=["%Change", "$Gain/Loss"], cmap="RdYlGn"
).set_properties(**{
    'text-align': 'center',
    'font-size': '12px'
})

st.dataframe(styled_df, use_container_width=True, height=350)

# -------------------------------------------------------------
# SUMMARY METRICS
# -------------------------------------------------------------
st.markdown("<h4 style='margin-bottom:4px;'>💼 Portfolio Summary</h4>", unsafe_allow_html=True)

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
