# Individual Company Stock Analysis
import streamlit as st
import pandas as pd
from datetime import date, timedelta
import ta
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import numpy as np
from typing import Tuple, Optional

# Page configuration
st.set_page_config(page_title="Alpha Stock 1", layout="wide")

# Create sidebar input for user configuration
st.sidebar.header('Configuration')
today = date.today()
DEFAULT_DAYS_BACK = 365
default_date = today - timedelta(days=DEFAULT_DAYS_BACK)


def get_input() -> Tuple[date, date, str]:
    """Retrieve user inputs from the sidebar."""
    # Widget commands must be outside cached functions
    start_date = st.sidebar.date_input("Start Date", default_date)
    end_date = st.sidebar.date_input("End Date", today)
    stock_symbol = st.sidebar.text_input("Stock Symbol", "PLTR").strip().upper()
    return start_date, end_date, stock_symbol


# Cache data functions for better performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch stock data and compute percentage changes."""
    try:
        end_adj = end + timedelta(days=1)
        df = yf.download(symbol, start=start, end=end_adj, progress=False)

        if df.empty:
            return df

        # Reset index to make Date a column and ensure proper formatting
        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        df['% Change'] = df['Close'].pct_change() * 100
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add MACD, CCI, and RSI indicators to the stock data."""
    if df.empty:
        return df

    try:
        # Ensure we're working with proper 1D series
        close_series = df['Close'].squeeze() if hasattr(df['Close'], 'squeeze') else df['Close']
        high_series = df['High'].squeeze() if hasattr(df['High'], 'squeeze') else df['High']
        low_series = df['Low'].squeeze() if hasattr(df['Low'], 'squeeze') else df['Low']

        # MACD
        macd = ta.trend.MACD(close_series)
        df['MACD'] = macd.macd()
        df['Signal'] = macd.macd_signal()
        df['MACD diff'] = macd.macd_diff()

        # CCI - Fixed: ensure we're passing 1D arrays
        df['CCI'] = ta.trend.cci(high_series, low_series, close_series, window=20)

        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(close_series).rsi()

        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        # Return dataframe without technical indicators if calculation fails
        return df


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_company_info(symbol: str) -> Tuple[str, str, dict]:
    """Fetch company information and key metrics."""
    try:
        ticker_data = yf.Ticker(symbol)
        info = ticker_data.info

        company_name = info.get('shortName', symbol)
        business_summary = info.get('longBusinessSummary', 'No company summary available.')

        return company_name, business_summary, info
    except Exception as e:
        st.warning(f"Could not fetch company information: {e}")
        return symbol, 'No company summary available.', {}


def plot_candlestick(df: pd.DataFrame, symbol: str) -> None:
    """Plot candlestick chart of stock data."""
    if df.empty:
        st.warning("No data available for candlestick chart.")
        return

    try:
        # Ensure the dataframe is sorted by date
        df_sorted = df.sort_index()

        fig = go.Figure(data=[go.Candlestick(
            x=df_sorted.index,
            open=df_sorted['Open'],
            high=df_sorted['High'],
            low=df_sorted['Low'],
            close=df_sorted['Close'],
            name='Price'
        )])

        fig.update_layout(
            title=f'{symbol} Candlestick Chart',
            yaxis_title="Price ($)",
            font=dict(size=12),
            xaxis_rangeslider_visible=False,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating candlestick chart: {e}")
        # Fallback to line chart
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')))
            fig.update_layout(
                title=f'{symbol} Price Chart (Fallback)',
                yaxis_title="Price ($)",
                font=dict(size=12),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.error("Unable to display any price chart.")


def create_indicator_chart(df: pd.DataFrame, symbol: str, indicator: str,
                           title: str, yaxis_title: str, **kwargs) -> go.Figure:
    """Create a standardized indicator chart."""
    fig = go.Figure()

    if indicator == 'RSI':
        # Check if RSI column exists and has data
        if 'RSI' in df.columns and not df['RSI'].isna().all():
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='blue')))
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            fig.update_layout(yaxis_range=[0, 100])
        else:
            fig.add_annotation(text="RSI data not available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    elif indicator == 'MACD':
        # Check if MACD columns exist
        if all(col in df.columns for col in ['MACD', 'Signal']) and not df['MACD'].isna().all():
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='red')))
        else:
            fig.add_annotation(text="MACD data not available", xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False)

    elif indicator == 'MACD_HIST':
        if 'MACD diff' in df.columns and not df['MACD diff'].isna().all():
            colors = np.where(df['MACD diff'] >= 0, 'green', 'red')
            fig.add_trace(go.Bar(x=df.index, y=df['MACD diff'], name='MACD Histogram',
                                 marker_color=colors))
        else:
            fig.add_annotation(text="MACD Histogram data not available", xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False)

    elif indicator == 'CCI':
        if 'CCI' in df.columns and not df['CCI'].isna().all():
            fig.add_trace(go.Scatter(x=df.index, y=df['CCI'], name='CCI', line=dict(color='purple')))
            fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Overbought (100)")
            fig.add_hline(y=-100, line_dash="dash", line_color="green", annotation_text="Oversold (-100)")
        else:
            fig.add_annotation(text="CCI data not available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    elif indicator == 'PCT_CHANGE':
        if '% Change' in df.columns and not df['% Change'].isna().all():
            colors = np.where(df['% Change'] >= 0, 'green', 'red')
            fig.add_trace(go.Bar(x=df.index, y=df['% Change'], name='Daily % Change',
                                 marker_color=colors))
        else:
            fig.add_annotation(text="Percentage Change data not available", xref="paper", yref="paper", x=0.5, y=0.5,
                               showarrow=False)

    fig.update_layout(
        title=f'{symbol} - {title}',
        yaxis_title=yaxis_title,
        font=dict(size=12),
        height=300
    )
    return fig


def plot_indicators(df: pd.DataFrame, symbol: str) -> None:
    """Plot technical indicator charts using a grid layout."""
    if df.empty:
        st.warning("No data available for technical indicators.")
        return

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_indicator_chart(df, symbol, 'RSI', 'RSI Chart', 'RSI'),
                        use_container_width=True)
        st.plotly_chart(create_indicator_chart(df, symbol, 'MACD', 'MACD Chart', 'MACD'),
                        use_container_width=True)
        st.plotly_chart(create_indicator_chart(df, symbol, 'PCT_CHANGE', 'Daily Percentage Change Chart', '% Change'),
                        use_container_width=True)

    with col2:
        st.plotly_chart(create_indicator_chart(df, symbol, 'MACD_HIST', 'MACD Histogram', 'MACD Difference'),
                        use_container_width=True)
        st.plotly_chart(create_indicator_chart(df, symbol, 'CCI', 'Commodity Channel Index (CCI) Chart', 'CCI'),
                        use_container_width=True)


def format_metric_value(value, metric_type: str = 'general') -> str:
    """Format metric values for display."""
    if not isinstance(value, (int, float)) or pd.isna(value):
        return 'N/A'

    if metric_type == 'currency':
        if value >= 1e9:
            return f"${value / 1e9:.2f}B"
        elif value >= 1e6:
            return f"${value / 1e6:.2f}M"
        else:
            return f"${value:,.0f}"
    elif metric_type == 'price':
        return f"${value:.2f}"
    elif metric_type == 'ratio':
        return f"{value:.2f}"
    else:
        return f"{value:.2f}"


def display_key_metrics(info: dict) -> None:
    """Display key financial metrics in columns."""
    try:
        col1, col2, col3, col4 = st.columns(4)

        metrics = [
            ('Current Price', info.get('currentPrice', info.get('regularMarketPrice')), 'price'),
            ('Previous Close', info.get('previousClose'), 'price'),
            ('Market Cap', info.get('marketCap'), 'currency'),
            ('P/E Ratio', info.get('trailingPE'), 'ratio')
        ]

        for i, (label, value, fmt_type) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                st.metric(label, format_metric_value(value, fmt_type))
    except Exception as e:
        st.warning("Could not display key statistics")


def display_upgrades_downgrades(symbol: str) -> None:
    """Display upgrades and downgrades information."""
    st.subheader("Upgrades & Downgrades")
    try:
        ticker_data = yf.Ticker(symbol)
        upgrades_downgrades = ticker_data.upgrades_downgrades

        if upgrades_downgrades is not None and not upgrades_downgrades.empty:
            st.dataframe(upgrades_downgrades, use_container_width=True)
        else:
            st.info("No recent upgrades or downgrades data available.")
    except Exception:
        st.info("Upgrades/downgrades data not available for this stock.")


def display_earnings_info(symbol: str) -> None:
    """Display earnings information."""
    st.subheader("Earnings Information")
    try:
        ticker_data = yf.Ticker(symbol)
        earnings_dates = ticker_data.earnings_dates

        if earnings_dates is not None and not earnings_dates.empty:
            # Get next earnings date
            future_earnings = earnings_dates[earnings_dates.index > pd.Timestamp.now()]
            if not future_earnings.empty:
                next_earnings_date = future_earnings.index[0].strftime("%d %b %Y")
                st.metric("Next Earnings Date", next_earnings_date)
            else:
                st.info("No future earnings dates scheduled.")

            # Show recent earnings
            st.write("Recent Earnings Dates:")
            st.dataframe(earnings_dates.head(10), use_container_width=True)
        else:
            st.info("No earnings dates data available.")
    except Exception:
        st.info("Earnings information not available for this stock.")


def display_daily_stock_data(df: pd.DataFrame, symbol: str) -> None:
    """Display daily stock data in a table at the bottom of the page."""
    st.subheader("Daily Stock Data")  # Removed symbol from title

    if df.empty:
        st.warning("No daily stock data available.")
        return

    try:
        # Create a copy of the dataframe for display
        display_df = df.copy()

        # Reset index to show Date as a column
        display_df = display_df.reset_index()

        # Sort by date in descending order (most recent first)
        if 'Date' in display_df.columns:
            display_df = display_df.sort_values('Date', ascending=False)
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')

        # Select columns for display (removed '% Change')
        columns_to_show = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in columns_to_show if col in display_df.columns]

        display_df = display_df[available_columns]

        # Display the raw dataframe without complex formatting
        st.write(f"Showing all {len(display_df)} trading days (most recent first):")

        # Use Streamlit's built-in dataframe with pagination
        st.dataframe(
            display_df,
            use_container_width=True,
            height=600
        )

    except Exception as e:
        st.error(f"Error displaying daily stock data: {e}")
        # Simple fallback - show raw data
        st.write("Raw data display:")
        st.dataframe(df, use_container_width=True)


# Main execution flow
def main():
    # Get user input first (widgets outside cached functions)
    start, end, symbol = get_input()
    st.sidebar.caption("ⓒ Franklin Chidi (FC) - MIT License")

    # Input validation
    if start >= end:
        st.error("Error: End date must be after start date.")
        return

    if not symbol:
        st.error("Error: Please enter a stock symbol.")
        return

    # Fetch and process data
    with st.spinner('Fetching stock data...'):
        df = get_stock_data(symbol, start, end)

    if df.empty:
        st.error(f"No data found for symbol '{symbol}'. Please check the symbol and date range.")
        return

    # Add technical indicators with error handling
    try:
        df = add_technical_indicators(df)
    except Exception as e:
        st.warning(f"Some technical indicators could not be calculated: {e}")
        # Continue with basic data even if technical indicators fail

    company_name, business_summary, info = get_company_info(symbol)

    # Display header and company info
    st.markdown(f"<h2 style='text-align: center; color: black;'>{company_name} ({symbol}) - Company Snapshot</h2>",
                unsafe_allow_html=True)

    st.subheader("Company Summary")
    st.write(business_summary)

    # Display metrics and charts
    display_key_metrics(info)

    st.subheader("Technical Analysis")
    plot_candlestick(df, symbol)
    plot_indicators(df, symbol)

    # Additional information
    display_upgrades_downgrades(symbol)
    display_earnings_info(symbol)

    # Display daily stock data at the bottom
    display_daily_stock_data(df, symbol)


if __name__ == "__main__":
    main()
