# Displays Today's top 50 winners and lossers
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from datetime import datetime
import time
import re
from zoneinfo import ZoneInfo

Eastern_time = datetime.now(ZoneInfo("America/New_York")) #Show Eastern Time

# ------------------------- CONFIG -------------------------
st.set_page_config(page_title="Alpha Gain-Loss Daily", layout="wide", page_icon="📊")
st.title("📊 Today's Gainers & Losers (Y! Finance)")

GAINERS_URL = "https://finance.yahoo.com/markets/stocks/gainers/?start=0&count=50"
LOSERS_URL = "https://finance.yahoo.com/markets/stocks/losers/?start=0&count=50"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}
# ------------------------- CACHED DATA FETCHING -------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_data():
    """Fetch both gainers and losers data in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_gainers = executor.submit(fetch_table, GAINERS_URL)
        future_losers = executor.submit(fetch_table, LOSERS_URL)

        gainers_df = future_gainers.result()
        losers_df = future_losers.result()

    return gainers_df, losers_df


def fetch_table(url: str):
    """Fetch and parse stock table from Yahoo Finance with multiple fallback strategies."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching {url}: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return pd.DataFrame()

    # Strategy 1: Try pandas read_html first (fastest)
    df = try_pandas_read_html(response.text)
    if not df.empty:
        return df

    # Strategy 2: Fallback to BeautifulSoup parsing
    df = try_beautifulsoup_parsing(response.text)
    if not df.empty:
        return df

    st.warning(f"Could not parse table from {url}")
    return pd.DataFrame()


def try_pandas_read_html(html_content: str) -> pd.DataFrame:
    """Attempt to parse tables using pandas read_html."""
    try:
        tables = pd.read_html(html_content, flavor='html5lib')
        for table in tables:
            if is_valid_stock_table(table):
                return table.head(50)  # Return early with valid table
    except Exception:
        pass
    return pd.DataFrame()


def try_beautifulsoup_parsing(html_content: str) -> pd.DataFrame:
    """Fallback parsing using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Try multiple table selectors
        table_selectors = [
            {'data-test': 'content-canvas'},
            {'class': re.compile('table')},
            {'class': re.compile('data-table')},
            {'class': 'W(100%)'},
        ]

        table = None
        for selector in table_selectors:
            table = soup.find('table', selector)
            if table:
                break

        if not table:
            return pd.DataFrame()

        return parse_html_table(table)

    except Exception as e:
        st.error(f"BeautifulSoup parsing failed: {e}")
        return pd.DataFrame()


def is_valid_stock_table(df: pd.DataFrame) -> bool:
    """Check if dataframe has expected stock table structure."""
    if df.empty or len(df.columns) < 4:
        return False

    required_keywords = ['symbol', 'price', 'change']
    column_keywords = ' '.join(str(col).lower() for col in df.columns)

    return all(keyword in column_keywords for keyword in required_keywords)


def parse_html_table(table) -> pd.DataFrame:
    """Parse HTML table element into DataFrame."""
    headers = []
    for th in table.find_all('th'):
        header_text = th.get_text(strip=True)
        headers.append(header_text)

    rows = []
    for tr in table.find_all('tr')[1:51]:  # Limit to 50 rows
        cells = [td.get_text(strip=True) for td in tr.find_all('td')]
        if cells and len(cells) >= len(headers):
            rows.append(cells[:len(headers)])  # Match cells to headers

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=headers).head(50)


# ------------------------- DATA PROCESSING -------------------------
def clean_table(df: pd.DataFrame, table_type: str = "gainers") -> pd.DataFrame:
    """Normalize, format, and sort the stock data."""
    if df.empty:
        return df

    # Create a copy to avoid modifying original
    df_clean = df.copy()
    df_clean.columns = [str(col).strip() for col in df_clean.columns]

    # Standardize column names with flexible matching
    column_mapping = create_column_mapping(df_clean.columns)
    df_clean = map_columns(df_clean, column_mapping)

    # Ensure we have required columns
    if 'Symbol' not in df_clean.columns:
        st.error(f"No Symbol column found in {table_type} data")
        return pd.DataFrame()

    # Clean and format data
    df_clean = clean_and_format_data(df_clean)

    # Sort data appropriately
    df_clean = sort_table(df_clean, table_type)

    # Add ranking column (1-50)
    df_clean = add_ranking_column(df_clean)

    return df_clean.head(50)


def add_ranking_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add ranking column as the first column."""
    if df.empty:
        return df

    # Create ranking column from 1 to number of rows
    df_ranked = df.copy()
    df_ranked.insert(0, '#', range(1, len(df_ranked) + 1))
    return df_ranked


def create_column_mapping(columns):
    """Create flexible column mapping based on available columns."""
    columns_lower = [str(col).lower() for col in columns]

    mapping = {}
    patterns = {
        'Symbol': [r'symbol', r'ticker'],
        'Name': [r'name', r'company'],
        'Price': [r'price', r'last'],
        'Change': [r'change$', r'change\s+\(', r'chg$'],
        'Change %': [r'change\s*%', r'%', r'percent'],
        'Volume': [r'volume$', r'vol$'],
        'Avg Vol (3M)': [r'avg', r'average', r'3m'],
        'Market Cap': [r'market.cap', r'mkt.cap'],
    }

    for std_col, regex_patterns in patterns.items():
        for pattern in regex_patterns:
            matches = [i for i, col in enumerate(columns_lower)
                       if re.search(pattern, col)]
            if matches:
                mapping[std_col] = columns[matches[0]]
                break

    return mapping


def map_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Map source columns to standardized column names."""
    result = pd.DataFrame()

    standard_columns = [
        'Symbol', 'Name', 'Price', 'Change', 'Change %',
        'Volume', 'Avg Vol (3M)', 'Market Cap'
    ]

    for std_col in standard_columns:
        if std_col in mapping:
            result[std_col] = df[mapping[std_col]]
        else:
            result[std_col] = pd.NA

    return result


def clean_and_format_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and format all data columns."""
    # Remove any completely empty rows
    df = df.dropna(how='all')

    # Format numeric columns
    if 'Price' in df.columns:
        df['Price'] = df['Price'].apply(format_currency)

    if 'Change' in df.columns:
        df['Change'] = df['Change'].apply(format_currency)

    if 'Change %' in df.columns:
        df['Change %'] = df['Change %'].apply(format_percentage)
        # Extract numeric value for sorting
        df['Change % Numeric'] = df['Change %'].apply(extract_numeric_percentage)

    # Format volume and market cap
    for col in ['Volume', 'Avg Vol (3M)', 'Market Cap']:
        if col in df.columns:
            df[col] = df[col].apply(format_large_number)

    return df


def format_currency(value):
    """Format currency values consistently."""
    if pd.isna(value):
        return "N/A"

    value_str = str(value).replace('$', '').replace(',', '').strip()

    try:
        num_value = float(value_str)
        return f"${num_value:,.2f}"
    except (ValueError, TypeError):
        return value


def format_percentage(value):
    """Format percentage values consistently."""
    if pd.isna(value):
        return "N/A"

    value_str = str(value).replace('%', '').replace('+', '').replace(',', '').strip()

    try:
        num_value = float(value_str)
        sign = "+" if num_value > 0 else ("-" if num_value < 0 else "")
        return f"{sign}{abs(num_value):.2f}%"
    except (ValueError, TypeError):
        return value


def extract_numeric_percentage(value):
    """Extract numeric value from percentage for sorting."""
    if pd.isna(value) or value == "N/A":
        return 0.0

    value_str = str(value).replace('%', '').replace('+', '').replace(',', '').strip()

    try:
        return float(value_str)
    except (ValueError, TypeError):
        return 0.0


def format_large_number(value):
    """Format large numbers (volume, market cap) for readability."""
    if pd.isna(value):
        return "N/A"

    value_str = str(value).upper()
    return value_str


def sort_table(df: pd.DataFrame, table_type: str) -> pd.DataFrame:
    """Sort table based on type (gainers/losers)."""
    if 'Change % Numeric' in df.columns:
        if table_type == "gainers":
            df = df.sort_values('Change % Numeric', ascending=False)
        else:  # losers
            df = df.sort_values('Change % Numeric', ascending=True)
        df = df.drop('Change % Numeric', axis=1)

    return df


# ------------------------- STREAMLIT UI -------------------------
def main():
    st.markdown("""
    <style>
    .stDataFrame {
        font-size: 14px;
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

    st.markdown("Real-time data from [Yahoo Finance](https://finance.yahoo.com/markets/stocks/gainers/)")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        auto_refresh = st.checkbox("Auto-refresh every 60 seconds", value=False)
        show_metrics = st.checkbox("Show Performance Metrics", value=True)

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    st.sidebar.caption(f"Refreshed: {Eastern_time.strftime('%Y-%m-%d at %-I:%M %p (Eastern)')}")
    st.sidebar.caption("ⓒ Franklin Chidi (FC) - MIT License")

    # Fetch data
    with st.spinner("Fetching latest market data..."):
        gainers_df, losers_df = fetch_all_data()

    # Display metrics
    if show_metrics and not gainers_df.empty and not losers_df.empty:
        display_performance_metrics(gainers_df, losers_df)

    # Display tables VERTICALLY (one under the other)
    display_stock_table(gainers_df, "📈 Top 50 Gainers", "gainers")

    # Add some spacing between tables
    st.markdown("<br>", unsafe_allow_html=True)

    display_stock_table(losers_df, "📉 Top 50 Losers", "losers")

    # Auto-refresh
    if auto_refresh:
        time.sleep(60)
        st.rerun()


def display_performance_metrics(gainers_df, losers_df):
    """Display performance metrics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not gainers_df.empty:
            st.metric("Gainers Found", len(gainers_df))

    with col2:
        if not losers_df.empty:
            st.metric("Losers Found", len(losers_df))

    with col3:
        if not gainers_df.empty and 'Change %' in gainers_df.columns:
            try:
                max_gain = max(extract_numeric_percentage(x) for x in gainers_df['Change %'].head(5))
                st.metric("Max Gain (Top 5)", f"{max_gain:.1f}%")
            except:
                st.metric("Max Gain", "N/A")

    with col4:
        if not losers_df.empty and 'Change %' in losers_df.columns:
            try:
                max_loss = min(extract_numeric_percentage(x) for x in losers_df['Change %'].head(5))
                st.metric("Max Loss (Top 5)", f"{max_loss:.1f}%")
            except:
                st.metric("Max Loss", "N/A")


def display_stock_table(df, title, table_type):
    """Display formatted stock table."""
    st.subheader(title)

    if df.empty:
        st.warning(f"No data found for {table_type}.")
        return

    cleaned_df = clean_table(df, table_type)

    if cleaned_df.empty:
        st.error(f"Could not process {table_type} data.")
        return

    # Apply styling for better visualization
    styled_df = cleaned_df.style.map(
        lambda x: 'color: green; font-weight: bold;' if isinstance(x, str) and '+' in x else
        'color: red; font-weight: bold;' if isinstance(x, str) and '-' in x else '',
        subset=['Change', 'Change %']
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=600  # Reduced height since tables are stacked
    )

    st.caption(f"Last updated: {Eastern_time.strftime('%Y-%m-%d at %-I:%M %p (Eastern)')}")


if __name__ == "__main__":
    main()
