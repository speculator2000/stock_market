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

Eastern_time = datetime.now(ZoneInfo("America/New_York")) #Show Eastern Time

# Default parameters
plt_style = "fivethirtyeight"
DEFAULT_WEIGHTS = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
DEFAULT_PORTFOLIO = "^GSPC, SYM, CRWV, KRYS, AAPL, AMZN, BBAI, BEAM, CELC, COGT, CRSP, DYN, GOOG, INTC, NVDA, META, PLTR, TSLA"

# ------------------------- CONFIG -------------------------
# Page configuration
st.set_page_config(page_title="Alpha Backtest", layout="wide")

# Title and Description
st.markdown("<h1 style='text-align: center; color: Grey;'>Optimal Portfolio Backtest Tool</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: Grey;'>Calculates optimal portfolio allocation and expected returns</h6>", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header('Configure Portfolio')

def get_input():
    """Get user input from sidebar."""
    default_start_date = date(date.today().year - 1, 12, 31)
    default_end_date = date.today()
    start_date = st.sidebar.date_input("Start Date", default_start_date)
    end_date = st.sidebar.date_input("End Date", default_end_date)
    portfolio = st.sidebar.text_input("Portfolio (format: NVDA, PLTR, GOOG)", DEFAULT_PORTFOLIO).upper()
    total_funds = st.sidebar.number_input("Total Funds to Invest", value=100000, step=5000, format='%d')
    return start_date, end_date, portfolio, total_funds

def fetch_data(symbols, start_date, end_date):
    """Fetch adjusted close price data for given symbols."""
    end_date_adj = end_date + timedelta(days=1)
    try:
        df = yf.download(symbols, start=start_date, end=end_date_adj, progress=False)['Close']
        # Handle single stock case
        if len(symbols) == 1:
            df = pd.DataFrame(df)
            df.columns = [symbols[0]]
    except Exception as e:
        st.warning(f"Error fetching data: {e}")
        return pd.DataFrame()
    return df

def calculate_weights(symbols):
    """Calculate equal weights for the portfolio."""
    num_stocks = len(symbols)
    return np.ones(num_stocks) / num_stocks

def plot_prices(df, title):
    """Plot stock prices."""
    fig = px.line(df)
    fig.update_layout(title=title, yaxis_title="", xaxis_title="", title_x=0.5, font=dict(family="Arial", size=11))
    st.plotly_chart(fig, use_container_width=True)

# User Input
start_date, end_date, portfolio, funds_to_invest = get_input()
symbols = [s.strip() for s in portfolio.split(',')]
funds_to_invest = max(funds_to_invest, 1)  # Ensure non-negative funds
st.sidebar.caption("ⓒ Franklin Chidi (FC) - MIT License")

# Fetch Data
if not symbols:
    st.error("Please enter at least one stock symbol.")
    st.stop()

df = fetch_data(symbols, start_date, end_date)

if df.empty or df.isnull().all().all():
    st.error("No valid stock data available. Please check the stock symbols or date range.")
    st.stop()

# Handle missing data
df = df.ffill().bfill()

# Normalize Data
df_normalized = df / df.iloc[0]

# Plot Stock Prices
#plot_prices(df, "Adjusted Daily Close Prices")
plot_prices(df_normalized, "Relative Daily Change Prices")

# Portfolio Analysis
returns = df.pct_change().dropna()

if returns.empty:
    st.error("Insufficient data to calculate returns. Please extend the date range.")
    st.stop()

cov_matrix_annual = returns.cov() * 252
weights = calculate_weights(symbols)

# Portfolio Metrics
portfolio_return = np.sum(returns.mean() * weights) * 252
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
portfolio_volatility = np.sqrt(portfolio_variance)

st.write(f"Expected Annual Return: {portfolio_return:.1%}")
st.write(f"Portfolio Volatility (Risk): {portfolio_volatility:.1%}")
st.write(f"Portfolio Variance - annualized: {portfolio_variance:.1%}")

# Portfolio Optimization
try:
    # Calculate the expected returns and the annualized sample covariance matrix of asset returns
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # Optimize for maximum Sharpe ratio
    ef = EfficientFrontier(mu, S)
    optimized_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    # Discrete Allocation
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=funds_to_invest)
    allocation, leftover = da.greedy_portfolio()

    # Results Summary
    df_weights = pd.DataFrame(list(cleaned_weights.items()), columns=['Stocks', 'Weight'])
    df_allocation = pd.DataFrame(list(allocation.items()), columns=['Stocks', '#Shares'])
    merged_df = df_weights.merge(df_allocation, on='Stocks', how='left')
    merged_df['Weight'] *= 100  # Convert to percentage

except Exception as e:
    st.error(f"Portfolio optimization failed: {e}")
    st.stop()

# Functions for portfolio calculations
def portfolio_profit(row):
    return row['#Shares'] * (row['End Price'] - row['Start Price'])

def profits_earned(row):
    return (row['End Price'] / row['Start Price'] - 1) * 100

def dollars_invested(row):
    return row['#Shares'] * row['Start Price']

def weight_multiplied_by_available_funds(row):
    return (row['Weight'] / 100) * funds_to_invest

def weighted_funds_divided_by_startPrice(row):
    return row['tmp_Investment'] / row['Start Price']

def dollar_returns(row):
    return row['Investment'] + row['$Gain/Loss']

# Get start and end prices
start_prices = df.iloc[0]
end_prices = df.iloc[-1]

# Create price dataframe with proper column names
price_df = pd.DataFrame({
    'Start Price': start_prices,
    'End Price': end_prices
})

# Merge with allocation data
merged_df.set_index('Stocks', inplace=True)
merged_df = merged_df.merge(price_df, left_index=True, right_index=True, how='left')

# Calculate portfolio metrics
merged_df['tmp_Investment'] = merged_df.apply(weight_multiplied_by_available_funds, axis=1)
merged_df['tmp_Shares to buy'] = merged_df.apply(weighted_funds_divided_by_startPrice, axis=1)
merged_df['#Shares'] = merged_df['tmp_Shares to buy'].apply(np.floor)
merged_df['Investment'] = merged_df.apply(dollars_invested, axis=1)
merged_df = merged_df.drop(columns=['tmp_Shares to buy', 'tmp_Investment'])
merged_df['%Change'] = merged_df.apply(profits_earned, axis=1)
merged_df['$Gain/Loss'] = merged_df.apply(portfolio_profit, axis=1)
merged_df['Total'] = merged_df.apply(dollar_returns, axis=1)

total_gain = merged_df['$Gain/Loss'].sum()
total_invested = merged_df['Investment'].sum()
total_percent_gain = total_gain / total_invested if total_invested > 0 else 0
leftover = funds_to_invest - total_invested
final_portfolio_value = total_invested + total_gain + leftover

st.write("Optimal Portfolio Allocation")
merged_df = merged_df.sort_values(by="Weight", ascending=False)

# Format the dataframe for display with proper column names
display_df = merged_df.copy()
display_df = display_df[['Weight', '#Shares', 'Start Price', 'End Price', 'Investment', '%Change', '$Gain/Loss', 'Total']]

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
st.write(f'Total funds invested: ${total_invested:,.0f}')
st.write(f'Uninvested balance: ${leftover:,.0f}')
st.write(f'Portfolio value increased by {total_percent_gain:.1%}, representing a gain of ${total_gain:,.0f}')
st.write(f'Final Portfolio value: ${final_portfolio_value:,.0f}')

#st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d - %H:%M:%S')}")
st.sidebar.caption(f"Updated: {Eastern_time.strftime('%Y-%m-%d at %I:%M %p (Eastern)')}")
