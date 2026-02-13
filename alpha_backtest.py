# Portfolio Alpha Backtest Tool
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

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="Alpha Backtest", layout="wide")

st.markdown("<h1 style='text-align: center; color: Grey;'>Optimal Portfolio Backtest Tool</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: Grey;'>Calculates optimal portfolio allocation and expected returns</h6>", unsafe_allow_html=True)

# ------------------------- SIDEBAR -------------------------
st.sidebar.header('Configure Portfolio')

# ------------------------- DEFAULTS -------------------------

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


@st.cache_data(ttl=300)  # 5-minute cache to protect API
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
        font=dict(family="Arial", size=11)
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------- USER INPUT -------------------------

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

plot_prices(df_normalized, "Relative Daily Change Prices")

# ------------------------- REFRESH CONTROLS -------------------------

if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0

#st.sidebar.markdown("### 🔄 Data Refresh")

manual_refresh = st.sidebar.button("Refresh")
auto_refresh = st.sidebar.checkbox("Enable Auto Refresh (Every 5 Hours)", value=False)

if manual_refresh:
    st.session_state.refresh_counter += 1
    st.rerun()

# Inject JavaScript auto-refresh (non-blocking, Streamlit Cloud safe)
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

# ------------------------- PORTFOLIO ANALYSIS -------------------------

returns = df.pct_change().dropna()

if returns.empty:
    st.error("Insufficient data to calculate returns.")
    st.stop()

cov_matrix_annual = returns.cov() * 252
weights = calculate_weights(symbols)

portfolio_return = np.sum(returns.mean() * weights) * 252
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
portfolio_volatility = np.sqrt(portfolio_variance)

st.write(f"Expected Annual Return: {portfolio_return:.1%}")
st.write(f"Portfolio Volatility (Risk): {portfolio_volatility:.1%}")
st.write(f"Portfolio Variance - annualized: {portfolio_variance:.1%}")

# ------------------------- OPTIMIZATION -------------------------

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

# ------------------------- CALCULATIONS -------------------------

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

# ------------------------- DISPLAY -------------------------

st.write("Optimal Portfolio Allocation")

merged_df = merged_df.sort_values(by="Weight", ascending=False)

display_df = merged_df[['Weight', '#Shares', 'Start Price', 'End Price',
                        'Investment', '%Change', '$Gain/Loss', 'Total']]

st.dataframe(display_df.style.format({
    "Weight": "{:.1f}%",
    "#Shares": "{:.0f}",
    "Start Price": "${:,.2f}",
    "End Price": "${:,.2f}",
    "Investment": "${:,.0f}",
    "%Change": "{:.1f}%",
    "Total": "${:,.0f}",
    "$Gain/Loss": "${:,.0f}"
}))

st.write(f"Total funds available: ${funds_to_invest:,.0f}")
st.write(f"Total funds invested: ${total_invested:,.0f}")
st.write(f"Uninvested balance: ${leftover:,.0f}")
st.write(f"Portfolio value increased by {total_percent_gain:.1%}, representing a gain of ${total_gain:,.0f}")
st.write(f"Final Portfolio value: ${final_portfolio_value:,.0f}")

st.sidebar.caption("ⓒ Franklin Chidi (FC) - MIT License")
st.sidebar.caption(f"Refreshed: {Eastern_time.strftime('%Y-%m-%d at %-I:%M %p (Eastern)')}")
