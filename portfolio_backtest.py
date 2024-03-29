# This is a Portfolio Analysis Tool which analyzes a portfolio of Stocks
import streamlit as st
from pandas_datareader import data as web
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
import yahoo_fin.stock_info as si
import plotly.express as px
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
yf.pdr_override()
plt.style.use('fivethirtyeight')
# Assign default weights to the portfolio
weights = np.array([0.20, 0.20, 0.20, 0.20, .20])
my_portfolio = "AMD, TSLA, MGM, AMZN, SIRI"
start_date = '2022-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

# Add a title and an image
st.markdown("<h1 style='text-align: center; color: Grey;'>Optimal Portfolio Backtest Tool</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: Grey;'>Calculates optimal portfolio allocation and expected portfolio return during chosen timeframe</h6>", unsafe_allow_html=True)

# Create a sidebar header
st.sidebar.header('Configure Portfolio')

# Create a dataframe to store the adjusted close price of the stocks
df = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()
df5 = pd.DataFrame()

# Create a function ot get the users input
today = date.today()
default_date = today - timedelta(days=365)
def get_input():
    with st.sidebar:
        start_date = st.date_input("Start Date",default_date)  # st.sidebar.text_input("Start Date", "2021/01/02")
        end_date = st.date_input("End Date")  # st.sidebar.text_input("End Date", str(datetime.now().strftime('%Y-%m-%d')))
        stock_symbol = st.sidebar.text_input("Enter Stocks in Portfolio (format: GE, AMZN, GOOG)", my_portfolio)
        funds_to_invest = st.sidebar.number_input("Total Funds to Invest", value=500000, step=50000, format='%d')
        return start_date, end_date, stock_symbol, funds_to_invest


# Create a function to get the proper company data and timeframe
def get_data(frm_symbol, data_source, start, end):

        for symbol in my_portfolio:
            df[symbol] = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)['Adj Close']
       #     df[symbol] = web.DataReader(symbol, data_source='yahoo', start=start_date, end=end_date)['Adj Close']
        # Get the date range
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        # set the start and end index rows both to zero
        start_row = 0
        end_row = 0

        # Start looking at the dataset from the top to see if the requested date range is in data set
        for i in range(0, len(df)):
            if start <= pd.to_datetime(df['Date'][i]):
                start_row = i
                break
        # Start from the bottom of the dataset and see if the requested date is greater ot equal tot the th
        for j in range(0, len(df)):
            if end >= pd.to_datetime(df['Date'][len(df)-1-j]):
                end_row = len(df) -1 - j
                break
        # set the index to the date
        df = df.set_index(pd.DatetimeIndex(df['Date'].values))

        return df.iloc[start_row:end_row +1, :]
# Get Updated Users Input from sidebar form

start, end, frm_symbol, funds_to_invest = get_input()
st.sidebar.caption("ⓒ Franklin Chidi (FC) - MIT License")
frm_symbol = frm_symbol.upper()
clean_form_Symbols = list(frm_symbol.split(', '))
#funds_to_invest.strip()
clean_funds_to_invest = int(funds_to_invest)
if clean_funds_to_invest < 1:
    clean_funds_to_invest = 1

# Get the data using cleaned symbols
for symbol in clean_form_Symbols:
    df[symbol] = pdr.get_data_yahoo(symbol, start=start, end=end)['Adj Close']
#    df[symbol] = web.DataReader(symbol, data_source='yahoo', start=start, end=end)['Adj Close']
    df4 = df.head(1).transpose()
    df5 = df.tail(1).transpose()

for c in df.columns.values:
    df2[c] = (df[c]/df[c].iloc[0])

# Update the calculated equal weight based on number of different stocks on the user form
Number_of_stocks = len(clean_form_Symbols)
weights_original = 1/Number_of_stocks   # Update the default weight for portfolio optimization
weights.resize(Number_of_stocks)        # resize the numpy array to current number of the portfolio
for i in weights:
    i = weights_original

# Display the Charts
fig = px.line(df)
fig.update_layout(
    title='Adjusted Daily Close Price of Stocks over Period',
    yaxis_title="", xaxis_title="",
    title_x=0.5,
    font=dict(family="Arial", size=11))
st.plotly_chart(fig,  use_container_width=True)

# Plot of Relative Daily Change between Stocks in Period
fig = px.line(df2)
fig.update_layout(
    title='Relative Daily Change between Stocks in Period',
    yaxis_title="", xaxis_title="",
    title_x=0.5,
    font=dict(family="Arial", size=11))
st.plotly_chart(fig,  use_container_width=True)

# Show the daily simple returns
returns = df.pct_change()

# Create the annualized covariance matrix
cov_matrix_annual = returns.cov() * 252
st.markdown("<h6 style='text-align: center; color: blue;'>Portfolio Covariance</h6>", unsafe_allow_html=True)
# st.write('Portfolio Covariance')
col1, col2, col3 = st.columns([1,2,1])
with col1:
    st.write("")
with col2:
    st.dataframe(cov_matrix_annual)
    #st.table(cov_matrix_annual)
with col3:
    st.write("")

# Calculate the portfolio variance
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
# st.write(port_variance)

# Calculate the portfolio volatility
port_volatility = np.sqrt(port_variance)
# st.write(port_volatility)

# Calculate the annual portfolio return
portforlioSimpleAnnualReturn = np.sum(returns.mean() * weights) * 252

# Show the expected annual return, volatility (risk) and variance
st.write('Expected Annual Return, if equal weights purchased:{: .1%}'.format(portforlioSimpleAnnualReturn))
st.write('Portfolio Volatility (Risk) - annualized:{: .1%}'.format(port_volatility))
st.write('Portfolio Variance - annualized:{: .1%}'.format(port_variance))


# Portfolio Optimization *********************************************************************
# Calculate the expected returns and the annualized sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for maximum Sharpe ratio
ef = EfficientFrontier(mu,S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
# Convert from dictionary to dataframe to display in table format
df_clean_weight_list = pd.DataFrame(list(cleaned_weights.items()),columns=['Stocks','Weight'])
# st.write('Recommended weighting:', df_clean_weight_list)
ef.portfolio_performance(verbose=True)
# Get the discrete allocation of each share per stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=clean_funds_to_invest)
# allocation, leftover = da.lp_portfolio()
allocation, leftover = da.greedy_portfolio()
# Convert from dictionary to dataframe to display in table format
df_allocation = pd.DataFrame(list(allocation.items()),columns=['Stocks','#Shares'])
# Set the index of both dataframes to same "stocks"
df_clean_weight_list.index = df_clean_weight_list['Stocks']
df_allocation.index = df_allocation['Stocks']
# Merge the two dataframes and then drop the duplicate column 'stocks' from old tables
df_merged1 = df_clean_weight_list.merge(df_allocation, left_index=True, right_index=True)
df_merged1 = df_merged1.drop(columns=['Stocks_x','Stocks_y'])
# Merge the head and tail from the stock price range into one dataframe
df_merged2 = pd.concat([df4,df5], axis=1)
df_merged2.columns = [start,end]  # Change the name of the start and End dates to the column
# Merge both dataframes to get one
df_portfolio_total = df_merged1.merge(df_merged2, left_index=True, right_index=True)


# Functions to calculate total dollars invested, % gains for each share, and total portfolio profits
def portfolio_profit(row):
    return row['#Shares'] * (row[end] - row[start])
def profits_earned(row):
    return (row[end] / row[start]-1)*100
def dollars_invested(row):
    return row['#Shares'] * row[start]
def weight_x_funds(row):
    return row['Weight'] * clean_funds_to_invest
def weighted_funds_divided_by_startPrice(row):
    return row['Funds Invested'] / row[start]
# Used .apply to use function and save the new calculated columns as Profits_
# _earned by each share, Gain/Loss in $ and total invested
# st.write(df_portfolio_total)



df_portfolio_total['Funds Invested'] = df_portfolio_total.apply(weight_x_funds, axis=1)
df_portfolio_total['Shares to buy'] = df_portfolio_total.apply(weighted_funds_divided_by_startPrice, axis=1)
df_portfolio_total['#Shares'] = df_portfolio_total['Shares to buy'].apply(np.floor)
df_portfolio_total = df_portfolio_total.drop(columns=['Shares to buy'])
df_portfolio_total['Gain/Loss'] = df_portfolio_total.apply(portfolio_profit, axis=1)
df_portfolio_total['%Change'] = df_portfolio_total.apply(profits_earned, axis=1)
df_total_invested_dollars = df_portfolio_total.apply(dollars_invested, axis=1)

total_gain = df_portfolio_total['Gain/Loss'].sum()  # Sum up the total dollar gain for portfolio
total_invested = df_total_invested_dollars.sum()    # Sum of total dollar invested
total_percent_gain = total_gain / total_invested    # Total portfolio percentage change
leftover_corrected = clean_funds_to_invest - total_invested
final_portfolio_value = total_invested + total_gain + leftover_corrected

# st.write("[Optimal Portfolio Allocation and Performance during Period]")
st.markdown("<h6 style='text-align: center; color: blue;'>Optimal Portfolio Allocation and Performance during Period</h6>", unsafe_allow_html=True)
df_portfolio_total['Weight'] = df_portfolio_total['Weight']*100

col1, col2, col3 = st.columns([.5,10,.5])
with col1:
    st.write("")
with col2:
    st.write(df_portfolio_total.style.format({'Weight': '{:.1f}%',
                                              '#Shares': '{:,.0f}',
                                              start: '${:,.2f}',
                                              end: '${:,.2f}',
                                              'Funds Invested': '${:,.0f}',
                                              'Gain/Loss': '${:,.0f}',
                                              '%Change': '{:.1f}%'}))
with col3:
    st.write("")

st.write('Invested a total of {:,.0f}'.format(total_invested) + ', leaving a cash balance of ${:.0f}'.format(leftover_corrected))
#st.write(weights_original)
st.write('Portfolio stocks changed by {:.1%}'.format(total_percent_gain) + ', for a total gain of ${:,.0f}'.format(total_gain))
st.write('Total Portfolio value is now ${:,.0f}'.format(final_portfolio_value))

