# Portfolio Backtest Tool
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
from alpha_portfolio import clean_form_Symbols, start

# Default parameters
plt_style = "fivethirtyeight"
default_date = '01/01/2025'
DEFAULT_WEIGHTS = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
DEFAULT_PORTFOLIO = "^GSPC, AAPL, AMZN, INTC, AMD, NVDA, META, TSLA, GOOG, ORCL"

# Title and Description
st.markdown("<h1 style='text-align: center; color: Grey;'>Optimal Portfolio Backtest Tool</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: Grey;'>Calculates optimal portfolio allocation and expected returns</h6>", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header('Configure Portfolio')

def get_input():
    """Get user input from sidebar."""
    default_date = date(date.today().year, 1, 1) #default_date = date.today() - timedelta(days=365)
    start_date = st.sidebar.date_input("Start Date", default_date)
    end_date = st.sidebar.date_input("End Date")
    portfolio = st.sidebar.text_input("Portfolio (format: NVDA, AMZN, GOOG)", DEFAULT_PORTFOLIO).upper()
    total_funds = st.sidebar.number_input("Total Funds to Invest", value=100000, step=5000, format='%d')
    return start_date, end_date, portfolio, total_funds

def fetch_data(symbols, start_date, end_date):
    """Fetch adjusted close price data for given symbols."""
    end_date = end_date + timedelta(days=1)
    df = pd.DataFrame()
    for symbol in symbols:
        try:
            df = yf.download(symbols, start=start_date, end=end_date, progress=False)['Adj Close']
        except Exception as e:(
            st.warning(f"Error fetching data for {symbol}: {e}"))
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
symbols = portfolio.split(', ')
funds_to_invest = max(funds_to_invest, 1)  # Ensure non-negative funds
st.sidebar.caption("ⓒ Franklin Chidi (FC) - MIT License")

# Fetch Data
df = fetch_data(symbols, start_date, end_date)

if df.empty:
    st.error("No valid stock data available. Please check the stock symbols or date range.")
    st.stop()

# Normalize Data
df_normalized = df / df.iloc[0]

# Plot Stock Prices
#plot_prices(df, "Adjusted Daily Close Prices")
plot_prices(df_normalized, "Relative Daily Change Prices")

# Portfolio Analysis
returns = df.pct_change()
cov_matrix_annual = returns.cov() * 252
weights = calculate_weights(symbols)

# Portfolio Metrics
portfolio_return = np.sum(returns.mean() * weights) * 252
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
portfolio_volatility = np.sqrt(portfolio_variance)

st.write(f"Expected Annual Return: {portfolio_return:.1%}")
st.write(f"Portfolio Volatility (Risk): {portfolio_volatility:.1%}")
st.write(f"Portfolio Variance - annualized: {portfolio_variance:.1%}")

# Portfolio Optimization++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

# Functions to calculate total dollars invested, % gains for each share, and total portfolio profits
def portfolio_profit(row):
    return row['#Shares'] * (row[end_date] - row[start_date])
def profits_earned(row):
    return (row[end_date] / row[start_date]-1)*100
def dollars_invested(row):
    return row['#Shares'] * row[start_date]
def weight_multiplied_by_available_funds(row):
    return (row['Weight']/100) * funds_to_invest
def weighted_funds_divided_by_startPrice(row):
    return row['tmp_Investment'] / row[start_date]
def dollar_returns(row):
    return row['Investment']+ row['$Gain/Loss']

df4 = df.head(1).transpose()
df5 = df.tail(1).transpose()
df4_start_date = df4.columns[0]
df5_end_date = df5.columns[0]
df_merged2 = pd.concat([df4,df5], axis=1)  # Merge both dataframes to get one
start_date = pd.to_datetime(df4_start_date)
start_date = start_date.date()   #Drop the timestamp
end_date = pd.to_datetime(df5_end_date)
end_date = end_date.date()   #Drop the timestamp
df_merged2.columns = [start_date, end_date]  # Change the name of the start and End dates to the column

merged_df.set_index('Stocks', inplace=True)
merged_df = pd.concat([merged_df,df_merged2], axis=1)


merged_df['tmp_Investment'] = merged_df.apply(weight_multiplied_by_available_funds, axis=1)
merged_df['tmp_Shares to buy'] = merged_df.apply(weighted_funds_divided_by_startPrice, axis=1)
merged_df['#Shares'] = merged_df['tmp_Shares to buy'].apply(np.floor)
merged_df['Investment'] = merged_df.apply(dollars_invested, axis=1)
merged_df = merged_df.drop(columns=['tmp_Shares to buy', 'tmp_Investment'])
#merged_df = merged_df.drop(columns=['tmp_Shares to buy', tmp_Investment'])
merged_df['%Change'] = merged_df.apply(profits_earned, axis=1)
merged_df['$Gain/Loss'] = merged_df.apply(portfolio_profit, axis=1)
merged_df['Total'] = merged_df.apply(dollar_returns, axis=1)

total_gain = merged_df['$Gain/Loss'].sum()  # Sum up the total dollar gain for portfolio
#df_total_invested_dollars = merged_df.apply(dollars_invested, axis=1)
#merged_df['Investment'].sum()
total_invested = merged_df['Investment'].sum() #df_total_invested_dollars.sum()    # Sum of total dollar invested
total_percent_gain = total_gain / total_invested    # Total portfolio percentage change
leftover = funds_to_invest - total_invested
final_portfolio_value = total_invested + total_gain + leftover

st.write("Optimal Portfolio Allocation")
merged_df = merged_df.sort_values(by="Weight", ascending=True)
st.dataframe(merged_df.style.format({"Weight": "{:.1f}%", "#Shares": "{:.0f}", start_date: '${:,.2f}',
                                              end_date: '${:,.2f}',
                                              'Investment': '${:,.0f}',
                                              '%Change': '{:.1f}%',
                                              'Total': '${:,.0f}',
                                              '$Gain/Loss': '${:,.0f}'}))

st.write(f"Total funds available: ${funds_to_invest:,.0f}")
st.write('Total funds invested: ${:,.0f}'.format(total_invested))
st.write('Uninvested balance: ${:,.0f}'.format(leftover))
st.write('Portfolio value increased by {:.1%}'.format(total_percent_gain) + ', representing a gain of ${:,.0f}'.format(total_gain))
st.write('Final Portfolio value: ${:,.0f}'.format(final_portfolio_value))

