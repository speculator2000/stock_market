import streamlit as st
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Set API keys
API_KEY = "JF4BRB377U85OERC"
ts = TimeSeries(key=API_KEY, output_format='pandas')

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Stock Analysis",
    layout="wide",
    page_icon="📈"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .positive {
        color: green;
        font-weight: bold;
    }
    .negative {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📊 Advanced Stock Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
Comprehensive stock analysis using Alpha Vantage API featuring price trends, volatility metrics, 
and risk-adjusted performance indicators.
""")

# Initialize session state for benchmark ticker
if 'benchmark_ticker' not in st.session_state:
    st.session_state.benchmark_ticker = 'SPY'

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Ticker input with popular examples
    ticker = st.text_input("Stock Ticker Symbol", 'GOOGL').upper()
    st.caption("Popular examples: AAPL, MSFT, TSLA, AMZN, GOOGL, META")

    # Date range with sensible defaults
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            'Start Date',
            value=datetime.now() - timedelta(days=365 * 2),  # 2 years default
            max_value=datetime.now() - timedelta(days=1)
        )
    with col2:
        end_date = st.date_input(
            'End Date',
            value=datetime.now(),
            max_value=datetime.now()
        )

    # Analysis options
    st.header("📊 Analysis Options")
    show_advanced_metrics = st.checkbox("Show Advanced Metrics", value=True)
    show_correlation = st.checkbox("Show Market Correlation", value=True)

    # Benchmark ticker with validation
    benchmark_input = st.text_input("Benchmark Ticker", st.session_state.benchmark_ticker).upper()
    if benchmark_input:
        st.session_state.benchmark_ticker = benchmark_input

    # Risk-free rate input (fallback when Quandl fails)
    rfr = st.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.5,
        step=0.1,
        help="Annual risk-free rate for Sharpe ratio calculation"
    ) / 100  # Convert to decimal

# Convert dates to string format
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")


# Enhanced data fetching with caching and error handling
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker_symbol):
    """Fetches daily stock data from Alpha Vantage with enhanced error handling."""
    try:
        data, metadata = ts.get_daily(symbol=ticker_symbol, outputsize='full')
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        # Rename columns for better readability
        column_mapping = {
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }
        data = data.rename(columns=column_mapping)

        return data, metadata
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {str(e)}")
        return None, None


# Calculate financial metrics
def calculate_metrics(price_data, benchmark_data, rfr=0.025):
    """Calculate comprehensive financial metrics."""
    metrics = {}

    # Basic returns
    price_data['Daily Return'] = price_data['Close'].pct_change(fill_method=None)
    benchmark_data['Daily Return'] = benchmark_data['Close'].pct_change(fill_method=None)

    # APR changes (annualized)
    price_data['APR Change'] = price_data['Daily Return'] * 100 * 252
    benchmark_data['APR Change'] = benchmark_data['Daily Return'] * 100 * 252

    # Filter data for selected date range
    price_filtered = price_data.loc[start_date_str:end_date_str]
    benchmark_filtered = benchmark_data.loc[start_date_str:end_date_str]

    if price_filtered.empty or benchmark_filtered.empty:
        return None, price_filtered, benchmark_filtered

    # Correlation
    metrics['Market Correlation'] = price_filtered['APR Change'].corr(benchmark_filtered['APR Change'])

    # Volatility (annualized)
    metrics['Stock Volatility'] = price_filtered['Daily Return'].std() * np.sqrt(252)
    metrics['Market Volatility'] = benchmark_filtered['Daily Return'].std() * np.sqrt(252)

    # Returns (annualized)
    metrics['Stock Return'] = price_filtered['Daily Return'].mean() * 252
    metrics['Market Return'] = benchmark_filtered['Daily Return'].mean() * 252

    # Beta calculation
    if metrics['Market Volatility'] != 0:
        metrics['Beta'] = metrics['Market Correlation'] * (metrics['Stock Volatility'] / metrics['Market Volatility'])
    else:
        metrics['Beta'] = 0

    # Alpha calculation
    metrics['Alpha'] = (metrics['Stock Return'] - rfr) - metrics['Beta'] * (metrics['Market Return'] - rfr)

    # Sharpe ratio
    metrics['Sharpe Ratio'] = (metrics['Stock Return'] - rfr) / metrics['Stock Volatility'] if metrics[
                                                                                                   'Stock Volatility'] != 0 else 0

    # Additional metrics
    metrics['Max Drawdown'] = calculate_max_drawdown(price_filtered['Close'])
    metrics['Total Return'] = (price_filtered['Close'].iloc[-1] / price_filtered['Close'].iloc[0] - 1) * 100

    return metrics, price_filtered, benchmark_filtered


def calculate_max_drawdown(prices):
    """Calculate maximum drawdown."""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min() * 100


def create_performance_chart(price_data, benchmark_data, ticker, benchmark_ticker):
    """Create interactive performance comparison chart."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Performance (Normalized)', 'Daily Returns'),
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )

    # Normalize prices to starting point = 100
    price_normalized = (price_data['Close'] / price_data['Close'].iloc[0]) * 100
    benchmark_normalized = (benchmark_data['Close'] / benchmark_data['Close'].iloc[0]) * 100

    # Price performance
    fig.add_trace(
        go.Scatter(x=price_data.index, y=price_normalized, name=f'{ticker} Price', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=benchmark_data.index, y=benchmark_normalized, name=f'{benchmark_ticker} Price',
                   line=dict(color='red')),
        row=1, col=1
    )

    # Daily returns
    fig.add_trace(
        go.Scatter(x=price_data.index, y=price_data['Daily Return'] * 100, name=f'{ticker} Returns',
                   line=dict(color='blue'), opacity=0.7),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=benchmark_data.index, y=benchmark_data['Daily Return'] * 100, name=f'{benchmark_ticker} Returns',
                   line=dict(color='red'), opacity=0.7),
        row=2, col=1
    )

    fig.update_layout(height=600, title_text=f"{ticker} vs {benchmark_ticker} Performance Analysis")
    fig.update_yaxes(title_text="Normalized Price (Base=100)", row=1, col=1)
    fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1)

    return fig


# Main application logic
def main():
    # Use session state for benchmark ticker to ensure it's always defined
    benchmark_ticker = st.session_state.benchmark_ticker

    # Fetch data
    with st.spinner("Fetching stock data..."):
        stock_data, stock_metadata = get_stock_data(ticker)
        benchmark_data, benchmark_metadata = get_stock_data(benchmark_ticker)

    # Handle data fetching errors
    if stock_data is None:
        st.error(f"❌ Could not fetch data for {ticker}. Please check the ticker symbol and try again.")
        st.info("🔍 Try popular tickers like: AAPL, MSFT, TSLA, GOOGL, AMZN, SPY")
        return

    # Handle benchmark data errors with fallback
    if benchmark_data is None:
        st.warning(f"⚠️ Could not fetch benchmark data for {benchmark_ticker}. Using SPY as default.")
        benchmark_data, benchmark_metadata = get_stock_data('SPY')
        if benchmark_data is None:
            st.error("❌ Could not fetch fallback benchmark data. Please check your connection and try again.")
            return
        benchmark_ticker = 'SPY'
        st.session_state.benchmark_ticker = 'SPY'  # Update session state

    # Calculate metrics
    metrics, price_filtered, benchmark_filtered = calculate_metrics(stock_data, benchmark_data, rfr)

    if metrics is None:
        st.error("❌ No data available for the selected date range. Please adjust your date range.")
        return

    # Display company information if available
    if stock_metadata:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ticker", ticker)
        with col2:
            st.metric("Time Zone", stock_metadata.get('6. Time Zone', 'N/A'))
        with col3:
            st.metric("Last Refresh", stock_metadata.get('7. Last Refreshed', 'N/A'))

    # Key metrics in columns
    st.subheader("📈 Key Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_price = price_filtered['Close'].iloc[-1]
        price_change = ((current_price / price_filtered['Close'].iloc[0]) - 1) * 100
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{price_change:+.2f}%"
        )

    with col2:
        st.metric("Total Return", f"{metrics['Total Return']:.2f}%")

    with col3:
        st.metric("Volatility", f"{metrics['Stock Volatility']:.2%}")

    with col4:
        sharpe_color = "green" if metrics['Sharpe Ratio'] > 0 else "red"
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")

    # Performance chart
    st.subheader("📊 Performance Analysis")
    performance_chart = create_performance_chart(price_filtered, benchmark_filtered, ticker, benchmark_ticker)
    st.plotly_chart(performance_chart, use_container_width=True)

    # Detailed metrics table
    st.subheader("🔍 Detailed Financial Metrics")

    metrics_data = {
        'Metric': [
            'Alpha', 'Beta', 'Sharpe Ratio', 'Market Correlation',
            'Total Return (%)', 'Annual Volatility (%)', 'Max Drawdown (%)',
            'Stock Return (Annualized %)', 'Market Return (Annualized %)'
        ],
        'Value': [
            f"{metrics['Alpha']:.4f}",
            f"{metrics['Beta']:.2f}",
            f"{metrics['Sharpe Ratio']:.2f}",
            f"{metrics['Market Correlation']:.2f}",
            f"{metrics['Total Return']:.2f}%",
            f"{metrics['Stock Volatility'] * 100:.2f}%",
            f"{metrics['Max Drawdown']:.2f}%",
            f"{metrics['Stock Return'] * 100:.2f}%",
            f"{metrics['Market Return'] * 100:.2f}%"
        ],
        'Description': [
            'Excess return over benchmark',
            'Sensitivity to market movements',
            'Risk-adjusted return',
            'Correlation with market',
            'Total period return',
            'Annualized volatility',
            'Maximum peak-to-trough decline',
            'Annualized stock return',
            'Annualized market return'
        ]
    }

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Raw data preview
    with st.expander("📋 View Raw Data Preview"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{ticker} Data (Last 10 rows)**")
            st.dataframe(price_filtered.tail(10))
        with col2:
            st.write(f"**{benchmark_ticker} Data (Last 10 rows)**")
            st.dataframe(benchmark_filtered.tail(10))

    # Footer
    st.markdown("---")
    st.caption(f"📊 Data Source: Alpha Vantage | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()