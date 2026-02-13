# -------------------------------------------------------------
# Today's Top 50 Gainers & Losers — Improved UI + 6‑Hour Refresh
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from datetime import datetime
import time
import re
from zoneinfo import ZoneInfo

Eastern_time = datetime.now(ZoneInfo("America/New_York"))

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Alpha Gain-Loss Daily",
    layout="wide",
    page_icon="📊"
)

# -------------------------------------------------------------
# GLOBAL COMPACT CSS
# -------------------------------------------------------------
st.markdown("""
    <style>
        .block-container {
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        h1, h2, h3, h4 {
            margin-top: 0.2rem;
            margin-bottom: 0.2rem;
        }
        .stMetric {
            padding: 0 !important;
        }
        hr {
            margin: 8px 0 !important;
        }
        .stDataFrame {
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.markdown("""
    <div style="text-align:center; padding: 20px 0 15px 0;">
        <h1 style="color:#4A4A4A; margin-bottom:8px; font-size:42px; font-weight:700;">
            📊 Today's Gainers & Losers (Yahoo Finance)
        </h1>
        <p style="color:#777; font-size:16px; margin-top:0;">
            Real‑time market data from <a href="https://finance.yahoo.com/markets/stocks/gainers/" target="_blank" style="color:#1f77b4; text-decoration:none;">
                Yahoo Finance
            </a>
        </p>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------
GAINERS_URL = "https://finance.yahoo.com/markets/stocks/gainers/?start=0&count=50"
LOSERS_URL = "https://finance.yahoo.com/markets/stocks/losers/?start=0&count=50"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html",
    "Accept-Language": "en-US,en;q=0.5",
}

# -------------------------------------------------------------
# FETCHING FUNCTIONS
# -------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_data():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_gainers = executor.submit(fetch_table, GAINERS_URL)
        future_losers = executor.submit(fetch_table, LOSERS_URL)
        return future_gainers.result(), future_losers.result()

def fetch_table(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except Exception as e:
        st.error(f"Error fetching {url}: {e}")
        return pd.DataFrame()

    df = try_pandas_read_html(response.text)
    if not df.empty:
        return df

    df = try_beautifulsoup_parsing(response.text)
    return df if not df.empty else pd.DataFrame()

def try_pandas_read_html(html):
    try:
        tables = pd.read_html(html, flavor="html5lib")
        for t in tables:
            if is_valid_stock_table(t):
                return t.head(50)
    except:
        pass
    return pd.DataFrame()

def try_beautifulsoup_parsing(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return pd.DataFrame()
        return parse_html_table(table)
    except:
        return pd.DataFrame()

def is_valid_stock_table(df):
    if df.empty or len(df.columns) < 4:
        return False
    keywords = ["symbol", "price", "change"]
    col_text = " ".join(str(c).lower() for c in df.columns)
    return all(k in col_text for k in keywords)

def parse_html_table(table):
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr")[1:51]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) >= len(headers):
            rows.append(cells[:len(headers)])
    return pd.DataFrame(rows, columns=headers)

# -------------------------------------------------------------
# CLEANING FUNCTIONS
# -------------------------------------------------------------
def clean_table(df, table_type="gainers"):
    if df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    mapping = create_column_mapping(df.columns)
    df = map_columns(df, mapping)

    if "Symbol" not in df.columns:
        return pd.DataFrame()

    df = clean_and_format_data(df)
    df = sort_table(df, table_type)
    df.insert(0, "#", range(1, len(df) + 1))
    return df.head(50)

def create_column_mapping(columns):
    cols_lower = [c.lower() for c in columns]
    mapping = {}
    patterns = {
        "Symbol": ["symbol", "ticker"],
        "Name": ["name", "company"],
        "Price": ["price", "last"],
        "Change": ["change"],
        "Change %": ["%"],
        "Volume": ["volume"],
        "Avg Vol (3M)": ["avg", "3m"],
        "Market Cap": ["cap"],
    }
    for std, pats in patterns.items():
        for p in pats:
            for i, col in enumerate(cols_lower):
                if p in col:
                    mapping[std] = columns[i]
                    break
    return mapping

def map_columns(df, mapping):
    cols = ["Symbol", "Name", "Price", "Change", "Change %", "Volume", "Avg Vol (3M)", "Market Cap"]
    out = pd.DataFrame()
    for c in cols:
        out[c] = df[mapping[c]] if c in mapping else pd.NA
    return out

def clean_and_format_data(df):
    df = df.dropna(how="all")
    df["Price"] = df["Price"].apply(format_currency)
    df["Change"] = df["Change"].apply(format_currency)
    df["Change %"] = df["Change %"].apply(format_percentage)
    df["Change % Numeric"] = df["Change %"].apply(extract_numeric_percentage)
    for col in ["Volume", "Avg Vol (3M)", "Market Cap"]:
        df[col] = df[col].apply(lambda x: str(x).upper() if pd.notna(x) else "N/A")
    return df

def format_currency(v):
    try:
        return f"${float(str(v).replace('$','').replace(',','')):,.2f}"
    except:
        return v

def format_percentage(v):
    try:
        num = float(str(v).replace('%','').replace('+','').replace(',',''))
        sign = "+" if num > 0 else "-" if num < 0 else ""
        return f"{sign}{abs(num):.2f}%"
    except:
        return v

def extract_numeric_percentage(v):
    try:
        return float(str(v).replace('%','').replace('+','').replace(',',''))
    except:
        return 0.0

def sort_table(df, table_type):
    if table_type == "gainers":
        return df.sort_values("Change % Numeric", ascending=False)
    return df.sort_values("Change % Numeric", ascending=True)

# -------------------------------------------------------------
# UI FUNCTIONS
# -------------------------------------------------------------
def display_stock_table(df, title, table_type):
    st.markdown(f"<h4 style='margin-bottom:4px;'>{title}</h4>", unsafe_allow_html=True)

    if df.empty:
        st.warning(f"No data found for {table_type}.")
        return

    cleaned = clean_table(df, table_type)
    if cleaned.empty:
        st.error(f"Could not process {table_type} data.")
        return

    styled = cleaned.style.map(
        lambda x: "color:green;font-weight:bold;" if isinstance(x, str) and "+" in x else
                  "color:red;font-weight:bold;" if isinstance(x, str) and "-" in x else "",
        subset=["Change", "Change %"]
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=480)
    st.caption(f"Last updated: {Eastern_time.strftime('%Y-%m-%d at %-I:%M %p (Eastern)')}")

def display_performance_metrics(gainers_df, losers_df):
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Gainers Found", len(gainers_df))
    col2.metric("Losers Found", len(losers_df))

    if not gainers_df.empty:
        top_gain = max(extract_numeric_percentage(x) for x in gainers_df["Change %"].head(5))
        col3.metric("Max Gain (Top 5)", f"{top_gain:.2f}%")

    if not losers_df.empty:
        top_loss = min(extract_numeric_percentage(x) for x in losers_df["Change %"].head(5))
        col4.metric("Max Loss (Top 5)", f"{top_loss:.2f}%")

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")

        refresh_option = st.selectbox(
            "Auto‑Refresh Interval",
            [
                "Off",
                "Every 60 seconds",
                "Every 5 minutes",
                "Every 30 minutes",
                "Every 1 hour",
                "Every 6 hours"
            ],
            index=0
        )

        show_metrics = st.checkbox("Show Performance Metrics", value=True)

        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()

        st.caption(f"Refreshed: {Eastern_time.strftime('%Y-%m-%d at %-I:%M %p (Eastern)')}")
        st.caption("ⓒ Franklin Chidi (FC) - MIT License")

    # Fetch data
    with st.spinner("Fetching latest market data..."):
        gainers_df, losers_df = fetch_all_data()

    # Metrics
    if show_metrics and not gainers_df.empty and not losers_df.empty:
        display_performance_metrics(gainers_df, losers_df)

    # Tables
    display_stock_table(gainers_df, "📈 Top 50 Gainers", "gainers")
    st.markdown("<hr>", unsafe_allow_html=True)
    display_stock_table(losers_df, "📉 Top 50 Losers", "losers")

    # Auto‑refresh
    refresh_seconds = {
        "Off": None,
        "Every 60 seconds": 60,
        "Every 5 minutes": 300,
        "Every 30 minutes": 1800,
        "Every 1 hour": 3600,
        "Every 6 hours": 21600
    }

    interval = refresh_seconds.get(refresh_option)

    if interval:
        st.markdown(
            f"""
            <script>
                setTimeout(function(){{
                    window.location.reload();
                }}, {interval * 1000});
            </script>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
