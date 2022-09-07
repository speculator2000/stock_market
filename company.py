# This is an analysis tool to be used for individual company stock
import streamlit as st
from pandas_datareader import data as web
import pandas as pd
from datetime import datetime
import yahoo_fin.stock_info as si
import requests
from bs4 import BeautifulSoup
import ta
from ta.utils import dropna
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

# Create a sidebar header
st.sidebar.header('Configuration')

df = pd.DataFrame()   # Create a dataframe to store the adjusted close price of the stocks
df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()
df5 = pd.DataFrame()
df6 = pd.DataFrame()
df_percent_chg = pd.DataFrame()
df_MACD = []


# Create a function to get user input
def get_input():
    with st.sidebar:
        start_date = st.text_input("Start Date", "2021/01/01")
        end_date = st.date_input("End Date")  # .sidebar.text_input("End Date", str(datetime.now().strftime('%Y-%m-%d')))
        stock_symbol = st.sidebar.text_input("Stock Symbol", "AMD")
        return start_date, end_date, stock_symbol


# create a function to get the proper company data and timeframe
def get_data(symbol, data_source, start, end):
        df[symbol] = web.DataReader(symbol, data_source='yahoo', start=start, end=end)
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
        # Start from the bottom of the dataset and see if the requested date is greater or equal to the the date
        for j in range(0, len(df)):
            if end >= pd.to_datetime(df['Date'][len(df)-1-j]):
                end_row = len(df) -1 - j
                break
        # set the index to the date
        df = df.set_index(pd.DatetimeIndex(df['Date'].values))
        return df.iloc[start_row:end_row +1, :]


# Get users input
start, end, symbol = get_input()
symbol = symbol.strip()
symbol = symbol.upper()
st.sidebar.caption("ⓒ Franklin Chidi (FC) - MIT License")

# Retrieve stock data
df = web.DataReader(symbol, data_source='yahoo', start=start, end=end)

# Calculate the Daily Percentage Returns
df2 = df['Adj Close']
df2 = round(df2.pct_change(), 4) * 100
df_percent_chg['% Change'] = df2.transpose()

# Get the Stock Statistics
stock_stats = si.get_stats(symbol)
try:
    df3 = si.get_analysts_info(symbol)["Earnings Estimate"]
except KeyError:
    print("No Earnings Statement")
try:
    df4 = si.get_analysts_info(symbol)["Earnings History"]
except KeyError:
    print("No Earnings History")

df = df.merge(df_percent_chg, left_index=True, right_index=True)
df = df.drop(columns=['Close'])

# MACD, CCI & RSI Computations
macd = ta.trend.MACD(close=df['Adj Close'])
df['MACD'] = macd.macd()
df['Signal'] = macd.macd_signal()
df['MACD diff'] = macd.macd_diff()
df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Adj Close'])
df['RSI'] = ta.momentum.RSIIndicator(df['Adj Close']).rsi()

# Sort dataframe to Decending Order
df.sort_values(by=['Date'], inplace=True, ascending=False)
df = dropna(df)

# Get subset of dataframe for MACD
df_MACD = df
df_MACD = df_MACD.drop(columns=["High", "Low", "Open", "Volume", "Adj Close", "% Change", "MACD diff", "CCI", "RSI"])

# Get the Name of the company
stock_company = f"https://finance.yahoo.com/quote/{symbol}"
soup = BeautifulSoup(requests.get(stock_company).text, "html.parser")
company_name = soup.h1.text.split('-')[0].strip()

# Add a title
st.header(company_name + """ Analysis""")

# get Company Summary on this Symbol
tickerData = yf.Ticker(symbol)
tickerData.info['longBusinessSummary']

# Get the next earnings statement date
try:
    earningsDate = si.get_next_earnings_date(symbol)
    strEarningsDate = earningsDate.strftime("%d %b %Y ")
except KeyError:
    strEarningsDate = "No Date"
except IndexError:
    earningsDate = 'null'
    strEarningsDate = 'No Date'
# Plot Candlestick Chart
df['Date'] = df.index
fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Adj Close'],
                name=symbol)])
fig.update_layout(
    title="=============>>> Earnings reported on: " + strEarningsDate + "<<<=============",
    yaxis_title="Price ($)",
    font=dict(family="Arial, monospace", size=12, color="blue"))
st.plotly_chart(fig,  use_container_width=True)

# Display the close price
st.subheader(symbol + "- Adjusted Close Price History\n")
df = df.drop(columns=["Date"])
st.write(df)

df['Date'] = df.index
# fig = go.Figure(data=[go.Scatter(x=df['Date'],y=df['RSI'])])
fig = px.line(x=df.index,y=df['RSI'])

fig.update_layout(
    title=symbol + ' - RSI Chart',
    yaxis_title="RSI",
    font=dict(family="Arial", size=12, color="black"))
fig.update_layout(yaxis_range=[0,100])
fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="70 RSI top line", annotation_position="top left")
fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="30 RSI bottom line", annotation_position="top left")
# fig.show()
st.plotly_chart(fig,  use_container_width=True)

# Plot the MACD Chart
fig = px.line(df_MACD)
fig.update_layout(
    title=symbol + ' - MACD Chart',
    yaxis_title="", xaxis_title="",
    font=dict(family="Arial", size=12, color="black"))
st.plotly_chart(fig,  use_container_width=True)

# Plot the MACD Diff Chart
fig = go.Figure(data=[go.Bar(x=df.index,y=df['MACD diff'])])
fig.update_layout(
    title=symbol + ' - MACD diff Chart',
    yaxis_title="MACD diff",
    font=dict(family="Arial", size=12, color="black"))
st.plotly_chart(fig,  use_container_width=True)

# Plot the CCI (Commodity Channel Index)  Chart
fig = go.Figure(data=[go.Scatter(x=df.index,y=df['CCI'])])
fig.update_layout(
    title=symbol + ' - Commodity Channel Index (CCI) Chart',
    yaxis_title="CCI",
    font=dict(family="Arial", size=12, color="black"))
# fig.add_hline(y=0, line_dash="dash", line_color="red")
st.plotly_chart(fig,  use_container_width=True)
# fig.show()

# Plot the daily % Change Chart
fig = go.Figure(data=[go.Bar(x=df.index,y=df['% Change'])])
fig.update_layout(
    title=symbol + ' - Daily Percentage Change Chart',
    yaxis_title="% Change",
    font=dict(family="Arial", size=12, color="black"))
st.plotly_chart(fig,  use_container_width=True)

# Display the Stock Statistics
st.header(symbol + " - Statistics")
st.write(stock_stats)
st.write(df3)
st.write(df4)
st.write(df5)

# Display the Volume
fig = go.Figure(data=[go.Scatter(x=df.index,y=df['Volume'])])
fig.update_layout(title=symbol + ' - Daily Volume', yaxis_title="", font=dict(family="Arial", size=11, color="black"))
st.plotly_chart(fig,  use_container_width=True)

