# This is an analysis tool to be used for a group of stocks
import requests
from bs4 import BeautifulSoup
import streamlit as st
from pandas_datareader import data as web
import pandas as pd
from datetime import datetime
import yahoo_fin.stock_info as si
import plotly.express as px
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Assign default weights to the portfolio
my_portfolio = "TSLA, ZM, AAPL, AMZN, GOOG, META, AMD, NVDA"
start_date = '2021-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
attribute_list = ['Market Cap', 'Trailing P/E', 'Forward P/E', 'PEG Ratio', 'Price/Sales', 'Price/Book', 'Enterprise Value/Revenue', 'Enterprise Value/EBITDA']
header_attribute_list = ['M Cap', 'T P/E', 'F P/E', 'PEG', 'P/S', 'P/B', 'EV/Rev', 'EV/EBITDA'] #To update final colunm names
attribute_list2 = ['Beta', '50-Day', '200-Day', '52 Week H', '52 Week L', '% Held by Inst', 'Short % of S', 'Profit Margin', 'Return on E']
header_attribute_list2 = ['Beta', '50d', '200d', '52W H', '52W L', '%Inst', '%Short', '%Profit', 'ROE']
risk_free = 0.015

# Add a title and an image
st.write("""
# Stock Portfolio Analysis - Chidi
""")

# Create a sidebar header
st.sidebar.header('User Input')
# Create a dataframe to store the adjusted close price of the stocks
df = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()
df6 = pd.DataFrame()


# Create a function to capture the default inputs for the start run
def get_input():
    start_date = st.sidebar.text_input("Start Date", "2021-01-02")
    end_date = st.sidebar.text_input("End Date", str(datetime.now().strftime('%Y-%m-%d')))
    stock_symbol = st.sidebar.text_input("Enter Stock Symbols (e.g. AAPL, GOOG, IBM)", my_portfolio)
    return start_date, end_date, stock_symbol


# Create a function to get the proper company data and timeframe
def get_data(frm_symbol, data_source, start, end):
        for symbol in my_portfolio:
            df[symbol] = web.DataReader(symbol, data_source='yahoo', start=start_date, end=end_date)['Adj Close']

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
        # Start from the bottom of the dataset and see if the requested date is greater or equal to the the dataset
        for j in range(0, len(df)):
            if end >= pd.to_datetime(df['Date'][len(df)-1-j]):
                end_row = len(df) -1 - j
                break
        # set the index to the date
        df = df.set_index(pd.DatetimeIndex(df['Date'].values))
        return df.iloc[start_row:end_row +1, :]


# Get Updated Users Input from sidebar form
start, end, frm_symbol = get_input()
frm_symbol = frm_symbol.upper()
clean_form_Symbols = list(frm_symbol.split(', '))

# Get the data using cleaned symbols
for symbol in clean_form_Symbols:
    df[symbol] = web.DataReader(symbol, data_source='yahoo', start=start, end=end)['Adj Close']
    df4 = df.tail(1).transpose()    # Gets the current prices to append to statistics dataframe
df4['Price'] = df4  # Current prices
df5 = df4['Price']

for c in df.columns.values:
    df2[c] = (df[c]/df[c].iloc[0])     # Calculates the relative daily change in a stock price

# Plot the Adjusted Closing Prices Chart
fig = px.line(df)
fig.update_layout(
    title='Adjusted Daily Close Price of Stocks over Period',
    yaxis_title="", xaxis_title="",
    font=dict(family="Arial", size=11))
st.plotly_chart(fig,  use_container_width=True)

# Plot of Relative Daily Change between Stocks in Period
fig = px.line(df2)
fig.update_layout(
    title='Relative Daily Change between Stocks in Period',
    yaxis_title="", xaxis_title="",
    font=dict(family="Arial", size=11))
st.plotly_chart(fig,  use_container_width=True)

df_Sortino = pd.DataFrame()
df_Sharpe = pd.DataFrame()
df_downside = pd.DataFrame()
df_upside = pd.DataFrame()
df_StnDev = pd.DataFrame()
df_len_Negatives = pd.DataFrame()
df_len_Positives = pd.DataFrame()
df_Earnings1 = pd.DataFrame()
df_Earnings2 = pd.DataFrame()

# Get Date of next earnings statement
for symbol in clean_form_Symbols:
    try:
        earningsPull = si.get_next_earnings_date(symbol)
        strEarningsDate = earningsPull.strftime("%d %b %Y ")
        df_Earnings1['Stock'] = [symbol]
        stock_company = f"https://finance.yahoo.com/quote/{symbol}"
        soup = BeautifulSoup(requests.get(stock_company).text, "html.parser")
        df_Earnings1['Company Name'] = soup.h1.text.split('-')[0].strip()
        df_Earnings1['Next Earnings Date'] = strEarningsDate
    except KeyError:
        df_Earnings1['Stock'] = [symbol]
        df_Earnings1['Next Earnings Date'] = "No Date"
    except IndexError:
        df_Earnings1['Stock'] = [symbol]
        df_Earnings1['Next Earnings Date'] = "No Date"
    df_Earnings2 = df_Earnings2.append(df_Earnings1, ignore_index=True)
df_Earnings2.index = df_Earnings2['Stock']
df_Earnings2 = df_Earnings2.drop(columns=['Stock'])

# Show the daily simple returns
returns = df.pct_change()
# Calculate negative returns for solving downside risk
df_len_Neg_Pos = pd.DataFrame()
df_len_Neg_Pos1 = pd.DataFrame()

for symbol in clean_form_Symbols:
    negative_returns = returns.loc[returns[symbol] < 0]
    positive_returns = returns.loc[returns[symbol] > 0]
    count_negatives = negative_returns.index
    count_negatives = len(count_negatives)
    count_positives = positive_returns.index
    count_positives = len(count_positives)
    df_len_Neg_Pos['Symbol'] = [symbol]
    df_len_Neg_Pos['Down'] = [count_negatives]
    df_len_Neg_Pos['Up'] = [count_positives]
    df_len_Neg_Pos.index = df_len_Neg_Pos['Symbol']
    df_len_Neg_Pos1 = df_len_Neg_Pos1.append(df_len_Neg_Pos, ignore_index=True)
df_len_Neg_Pos1.index = df_len_Neg_Pos1['Symbol']
df_len_Neg_Pos1 = df_len_Neg_Pos1.drop(columns=['Symbol'])


def percent_down(row):
    return row['Down'] / (row['Down'] + row['Up']) * 100


def percent_up(row):
    return row['Up'] / (row['Down'] + row['Up']) * 100


df_len_Neg_Pos1['Down%'] = df_len_Neg_Pos1.apply(percent_down, axis=1)
df_len_Neg_Pos1['Up%'] = df_len_Neg_Pos1.apply(percent_up, axis=1)

#  Prints the first dataframe with company name etc
df_len_Neg_Pos1 = df_len_Neg_Pos1.round()
df_len_Neg_Pos1 = df_len_Neg_Pos1.astype(int)  # Sets the dataframe to zero decimals
# df_len_Neg_Pos1 = df_len_Neg_Pos1.style.set_precision(0)

df_Earnings2['Down % in Period'] = df_len_Neg_Pos1['Down%']
df_Earnings2['Up % in Period'] = df_len_Neg_Pos1['Up%']
st.write(df_Earnings2)

#  Calculate expected return and std dev of downside returns
expected_return = returns[symbol].mean()
df_downside['Downside'] = negative_returns.std()
df_upside['Upside'] = positive_returns.std()
df_StnDev['Std Dev'] = returns.std()
#  Calculate the sortino ratio
df_Sortino['Sortino'] = (expected_return - risk_free) / negative_returns.std()
df_Sharpe['Sharpe'] = (expected_return - risk_free) / returns.std()

# Defined dataframes for write output
df2 = pd.DataFrame()
df2_3 = pd.DataFrame()
stocks_list = {}

for stox in clean_form_Symbols:
    temp = si.get_stats(stox)
    temp = temp.iloc[:, :2]     # Pick up 2
    stocks_list[stox] = temp
# combine all the stats valuation tables into a single data frame
df2_1 = pd.concat(stocks_list)
df2_1 = df2_1.reset_index()

for attributes2 in attribute_list2:
    df2_2 = df2_1[df2_1.Attribute.str.contains(attributes2)]
    df2_2.index = df2_2['level_0']
    df2_2.columns = ["level_0", "level_1", "Attribute", attributes2]
    df2_2 = df2_2.drop(columns=["level_0", "level_1", "Attribute"])
    df2_3 = pd.concat([df2_3,df2_2],axis=1)
df2_3.columns = [header_attribute_list2]

# Get data in the current column for each stock's valuation table
dow_stats = {}
for ticker in clean_form_Symbols:
    temp = si.get_stats_valuation(ticker)
    temp = temp.iloc[:, :2]     # Pick up 2
    temp.columns = ["Attribute", "Recent"]
    dow_stats[ticker] = temp
# combine all the stats valuation tables into a single data frame
combined_stats = pd.concat(dow_stats)
combined_stats = combined_stats.reset_index()
del combined_stats["level_1"]
combined_stats.columns = ["Ticker", "Attribute", 'Market Cap1']

# Create the individual tables and combine unto one
for attributes in attribute_list:
    df1 = combined_stats[combined_stats.Attribute.str.contains(attributes)]
    df1.index = df1['Ticker']
    df1.columns = ["Ticker", "Attribute", attributes]
    df1 = df1.drop(columns=['Ticker', 'Attribute'])
    df2 = pd.concat([df2,df1],axis=1)
df2.columns = [header_attribute_list]

df2_3.to_csv('file1.csv')

df_calculated2 = pd.read_csv('file1.csv')
df_calculated2 = df_calculated2.drop(df_calculated2.index[0])
df_calculated2.index = df_calculated2['Unnamed: 0']
df_calculated2 = df_calculated2.drop(columns=['Unnamed: 0'])

# This strips out the % sign from the dataframe in these four columns
df_calculated2['%Inst'] = list(map(lambda x: x[:-1], df_calculated2['%Inst'].values))
df_calculated2['%Short'] = list(map(lambda x: x[:-1], df_calculated2['%Short'].values))
df_calculated2['%Profit'] = list(map(lambda x: x[:-1], df_calculated2['%Profit'].values))
df_calculated2['ROE'] = list(map(lambda x: x[:-1], df_calculated2['ROE'].values))
# Now to change the format to float
df_calculated2['%Inst'] = [float(x) for x in df_calculated2['%Inst'].values]
df_calculated2['%Short'] = [float(x) for x in df_calculated2['%Short'].values]
df_calculated2['%Profit'] = [float(x) for x in df_calculated2['%Profit'].values]
df_calculated2['ROE'] = [float(x) for x in df_calculated2['ROE'].values]
df_calculated2 = pd.concat([df5, df_calculated2], axis=1)

df_calculated3 = pd.DataFrame()
df_calculated3['Price'] = df_calculated2['Price']
df_calculated3['Sortino'] = df_Sortino['Sortino']
df_calculated3['Sharpe'] = df_Sharpe['Sharpe']
df_calculated3['Std Dev'] = df_StnDev['Std Dev']*100
df_calculated3['Dside'] = df_downside['Downside']*100
df_calculated3['Uside'] = df_upside['Upside']*100
df_calculated3['Beta'] = df_calculated2['Beta']
df_calculated3['P-52H%'] = ((df_calculated2['Price'] / df_calculated2['52W H'])-1)*100
df_calculated3['P-52L%'] = ((df_calculated2['Price'] / df_calculated2['52W L'])-1)*100
df_calculated3['50-200%'] = ((df_calculated2['50d'] / df_calculated2['200d'])-1)*100

# Rearrange a few colunms between the two calculated dataframes
df_calculated3['%Inst'] = df_calculated2['%Inst']
df_calculated3['%Short'] = df_calculated2['%Short']
df_calculated3['%Profit'] = df_calculated2['%Profit']
df_calculated3['ROE'] = df_calculated2['ROE']
# next we move columns from table 3 to 2 before dropping from 3
df_calculated2['P-52H%'] = df_calculated3['P-52H%']
df_calculated2['P-52L%'] = df_calculated3['P-52L%']
df_calculated2['50-200%'] = df_calculated3['50-200%']
# Now we drop the columns
df_calculated2 = df_calculated2.drop(columns=['Beta', 'ROE', '%Inst', '%Short', '%Profit'])
df_calculated3 = df_calculated3.drop(columns=['Price', 'P-52H%', 'P-52L%', '50-200%'])
df_calculated2 = df_calculated2[['Price', '52W H', '52W L', 'P-52H%', 'P-52L%', '50d', '200d', '50-200%']]
#   Print the dataframes to webpage
st.dataframe(df_calculated2.style.set_precision(2))
st.dataframe(df_calculated3.style.set_precision(2))
st.write(df2)

