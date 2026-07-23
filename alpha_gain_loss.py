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

# =============================================================================
# DESIGN SYSTEM
# -----------------------------------------------------------------------------
# Same research-desk aesthetic used across the other apps in this suite: ink
# slate + ledger ivory, deep emerald / antique gold accents, Fraunces for
# display type, Inter for body text, IBM Plex Mono for figures.
# =============================================================================

PALETTE = {
    "ink": "#2B3B50",        # soft slate navy — sidebar, headings
    "ink_2": "#374B65",      # secondary ink surface
    "paper": "#F6F4EE",      # warm ivory — page background
    "paper_2": "#EFEBE0",    # card / metric surface
    "rule": "rgba(43,59,80,0.10)",   # hairline dividers
    "text": "#33404F",       # body text on paper
    "muted": "#697787",      # secondary text
    "paper_text": "#D9D4C7", # text on ink surfaces
    "emerald": "#33604F",    # primary accent — gainers, gains
    "gold": "#B0925F",       # secondary accent — highlights, rules
    "burgundy": "#8A4A4A",   # losers, risk / loss accent
}


def inject_design_system():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        .stApp {{
            background: {PALETTE["paper"]};
            color: {PALETTE["text"]};
            font-size: 0.92rem;
        }}

        .block-container {{
            padding-top: 1.4rem !important;
            padding-bottom: 1.3rem !important;
            max-width: 1300px;
        }}

        /* ---------- Typography ---------- */
        h1, h2, h3, h4 {{
            font-family: 'Fraunces', serif !important;
            color: {PALETTE["ink"]} !important;
            font-weight: 500 !important;
            letter-spacing: -0.01em;
            margin-top: 0.2rem;
            margin-bottom: 0.2rem;
        }}
        h4 {{
            border-bottom: 1px solid {PALETTE["rule"]};
            padding-bottom: 0.25rem;
            margin-top: 1rem !important;
            font-size: 1.05rem !important;
        }}
        .eyebrow {{
            display: block;
            font-family: 'Inter', sans-serif;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            color: {PALETTE["gold"]};
            margin-bottom: 0.1rem;
        }}

        /* ---------- Masthead ---------- */
        .masthead {{
            border-top: 2px solid {PALETTE["ink"]};
            border-bottom: 1px solid {PALETTE["rule"]};
            padding: 0.5rem 0 0.6rem 0;
            margin-bottom: 0.9rem;
            text-align: center;
        }}
        .masthead .eyebrow {{ margin-bottom: 0.2rem; text-align: center; }}
        .masthead h1 {{
            font-size: 2.0rem !important;
            margin: 0 !important;
            line-height: 1.15;
        }}
        .masthead .dek {{
            font-family: 'Inter', sans-serif;
            color: {PALETTE["muted"]};
            font-size: 0.82rem;
            margin-top: 0.2rem;
        }}
        .masthead .dek a {{ color: {PALETTE["ink"]}; text-decoration: none; border-bottom: 1px solid {PALETTE["gold"]}; }}

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {{
            background: {PALETTE["ink"]};
        }}
        [data-testid="stSidebar"] * {{
            color: {PALETTE["paper_text"]} !important;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            font-family: 'Fraunces', serif !important;
            font-weight: 500 !important;
            color: {PALETTE["paper_text"]} !important;
            border-bottom: 1px solid rgba(217,212,199,0.14);
            padding-bottom: 0.3rem;
            font-size: 1.05rem !important;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: rgba(217,212,199,0.12) !important;
            margin: 0.6rem 0 !important;
        }}
        [data-testid="stSidebar"] label {{ color: {PALETTE["paper_text"]} !important; opacity: 0.8; }}

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: {PALETTE["ink_2"]} !important;
            border: 1px solid rgba(217,212,199,0.16) !important;
            color: {PALETTE["paper_text"]} !important;
            border-radius: 4px !important;
        }}

        /* ---------- Buttons ---------- */
        .stButton > button, button[kind="primary"] {{
            background: {PALETTE["emerald"]} !important;
            color: {PALETTE["paper_text"]} !important;
            border: 1px solid {PALETTE["emerald"]} !important;
            border-radius: 4px !important;
            font-weight: 500 !important;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            font-size: 0.74rem !important;
            padding: 0.35rem 0.85rem !important;
        }}
        .stButton > button:hover, button[kind="primary"]:hover {{
            background: {PALETTE["ink"]} !important;
            border-color: {PALETTE["gold"]} !important;
            color: {PALETTE["gold"]} !important;
        }}

        /* ---------- Metrics ---------- */
        [data-testid="stMetric"] {{
            background: {PALETTE["paper_2"]};
            border: 1px solid {PALETTE["rule"]};
            border-radius: 6px;
            padding: 0.5rem 0.65rem 0.4rem 0.65rem !important;
        }}
        [data-testid="stMetricLabel"] {{
            font-family: 'Inter', sans-serif !important;
            font-size: 0.66rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {PALETTE["muted"]} !important;
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'IBM Plex Mono', monospace !important;
            color: {PALETTE["ink"]} !important;
            font-weight: 500 !important;
            font-size: 1.25rem !important;
        }}

        /* ---------- Dataframes ---------- */
        [data-testid="stDataFrame"] {{
            border: 1px solid {PALETTE["rule"]};
            border-radius: 6px;
            overflow: hidden;
            font-size: 0.85rem;
        }}

        /* ---------- Rules ---------- */
        hr {{ border-color: {PALETTE["rule"]} !important; margin: 0.6rem 0 !important; }}

        /* ---------- General compaction ---------- */
        div[data-testid="stVerticalBlock"] {{ gap: 0.4rem; }}
        div[data-testid="stHorizontalBlock"] {{ gap: 0.6rem; }}
        .element-container {{ margin-bottom: 0.1rem !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def masthead(eyebrow, title, dek):
    st.markdown(
        f"""
        <div class="masthead">
            <span class="eyebrow">{eyebrow}</span>
            <h1>{title}</h1>
            <div class="dek">{dek}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_design_system()

masthead(
    eyebrow="Alpha Desk &middot; Daily Movers",
    title="Today's Gainers &amp; Losers",
    dek='Real-time market data from <a href="https://finance.yahoo.com/markets/stocks/gainers/" target="_blank">Yahoo Finance</a>',
)

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
def display_stock_table(df, title, eyebrow, table_type):
    st.markdown(
        f"""<div class="section-head"><span class="eyebrow">{eyebrow}</span><h4>{title}</h4></div>""",
        unsafe_allow_html=True
    )

    if df.empty:
        st.warning(f"No data found for {table_type}.")
        return

    cleaned = clean_table(df, table_type)
    if cleaned.empty:
        st.error(f"Could not process {table_type} data.")
        return

    styled = cleaned.style.map(
        lambda x: f"color:{PALETTE['emerald']};font-weight:600;" if isinstance(x, str) and "+" in x else
                  f"color:{PALETTE['burgundy']};font-weight:600;" if isinstance(x, str) and "-" in x else "",
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
        st.markdown('<span class="eyebrow">Desk Setup</span>', unsafe_allow_html=True)
        st.header("Controls")

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

        if st.button("Refresh Now"):
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
    display_stock_table(gainers_df, "Top 50 Gainers", "Advancing", "gainers")
    st.markdown("<hr>", unsafe_allow_html=True)
    display_stock_table(losers_df, "Top 50 Losers", "Declining", "losers")

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
