import streamlit as st
from datetime import datetime, date, timedelta
import pandas as pd
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Page configuration
st.set_page_config(page_title="Alpha Portfolio", layout="wide")

# Assign default weights to the portfolio
my_portfolio = "AMD, NVDA, META, TSLA, GOOG, AAPL, AMZN, ORCL, INTC"
start_date = '2024-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

risk_free = 0.04 / 252

# Add a title
st.markdown("<h1 style='text-align: center; color: Grey;'>Portfolio Analysis</h2>", unsafe_allow_html=True)

# Create a sidebar header
st.sidebar.header('Configuration')
# Get the user inputs for start date, end date, and stock symbols
today = date.today()
default_date = today - timedelta(days=365)


def get_input():
    """Collect user input: start date, end date, and stock symbols."""
    with st.sidebar:
        start_date = st.date_input("Start Date", default_date)
        end_date = st.date_input("End Date")
        stock_symbol = st.sidebar.text_input("Enter Stock Symbols (e.g. AAPL, GOOG, IBM)", my_portfolio)
        return start_date, end_date, stock_symbol

# Fetch data from yfinance using the provided date range and stock symbols
def get_stock_data(symbols, start, end):
    """Fetch stock data from yfinance and return a DataFrame with adjusted close prices."""
    try:
        df = yf.download(symbols, start=start, end=end)['Adj Close']
    except KeyError:
        st.error("Could not retrieve stock data for one or more symbols. Please check the ticker symbols.")
        df = pd.DataFrame()  # Return an empty DataFrame if data fetching fails
    return df

# Get Updated Users Input from sidebar form
start, end, frm_symbol = get_input()
frm_symbol = frm_symbol.upper()
st.sidebar.caption("ⓒ Franklin Chidi (FC) - MIT License")
clean_form_Symbols = list(frm_symbol.split(', '))

# Fetch stock data
df = get_stock_data(clean_form_Symbols, start, end)

if df.empty:
    st.error("No stock data was fetched. Please check your input symbols.")
else:
    # Continue processing only if the data is successfully fetched
    st.write(df)
    df4 = df.tail(1).transpose()  # Get the current prices
    df4['Price'] = df4.iloc[:, 0] #df4[0]  # Adding a 'Price' column
    df5 = df4['Price']

    # Calculate relative daily change in stock prices
    df2 = df.div(df.iloc[0])  # Relative daily change in stock price

    # Plot the Adjusted Closing Prices Chart
    fig = px.line(df, title='Adjusted Daily Close Price of Stocks over Period',
                  labels={'value': 'Price', 'index': 'Date'})
    st.plotly_chart(fig, use_container_width=True)

    # Plot of Relative Daily Change between Stocks in Period
    fig2 = px.line(df2, title='Relative Daily Change between Stocks in Period',
                   labels={'value': 'Relative Change', 'index': 'Date'})
    st.plotly_chart(fig2, use_container_width=True)

    # Calculate returns and downside/upside standard deviation
    returns = df.pct_change()
    negative_returns = returns[returns < 0]
    positive_returns = returns[returns > 0]

    # Calculate Sortino and Sharpe ratios
    sortino_ratio = (returns.mean() - risk_free) / negative_returns.std()
    sharpe_ratio = (returns.mean() - risk_free) / returns.std()

    # Create DataFrames to display Sortino and Sharpe ratios along with standard deviation
    df_calculated=pd.DataFrame()
    df_calculated['Sortino'] = pd.DataFrame(sortino_ratio, columns=['Sortino'])
    df_calculated['Sharpe'] = pd.DataFrame(sharpe_ratio, columns=['Sharpe'])
    df_calculated['StnDev'] = pd.DataFrame(returns.std() * 100, columns=['Std Dev'])
    df_calculated['Downside'] = pd.DataFrame(negative_returns.std() * 100, columns=['Downside'])
    df_calculated['Upside'] = pd.DataFrame(positive_returns.std() * 100, columns=['Upside'])

    # Show calculated metrics in a clean format
    st.write("### Calculated Financial Metrics", df_calculated)
