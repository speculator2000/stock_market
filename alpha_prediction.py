"""
Top Gainers Stock Picking Pipeline
-----------------------------------
Fetches today's top gainers from Yahoo Finance, cleans and validates the
data, screens it against liquidity/momentum/volatility criteria, enriches
the survivors with fundamentals, scores them, and prints a ranked shortlist.

This is a quantitative screen for educational purposes — not financial advice.
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GAINERS_URL = "https://finance.yahoo.com/markets/stocks/gainers/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}

# Column names we want in the cleaned output, and the keyword(s) used to
# find each one in whatever header text the page actually renders. Built
# dynamically instead of assuming a fixed column order/count, since a
# mismatch there used to silently misalign every row.
COLUMN_PATTERNS: Dict[str, List[str]] = {
    "Symbol": ["symbol", "ticker"],
    "Name": ["name", "company"],
    "Price": ["price", "last"],
    "Change": ["change"],
    "Change %": ["change %", "% change", "change%"],
    "Volume": ["volume"],
    "Avg Vol (3M)": ["avg vol", "3m"],
    "Market Cap": ["market cap"],
    "P/E Ratio (TTM)": ["p/e"],
    "52 Wk Change %": ["52 wk change", "52wk change"],
    "52 Wk Range": ["52 wk range", "52wk range"],
}

MAX_FUNDAMENTALS_WORKERS = 6
FUNDAMENTALS_RETRY_ATTEMPTS = 2
FUNDAMENTALS_RETRY_DELAY_SECONDS = 1.0


def _build_session() -> requests.Session:
    """A requests session with automatic retries for transient network errors."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# --- Module 1: Data Acquisition & Parsing ---
def _map_headers(header_texts: List[str]) -> Dict[str, int]:
    """Match the page's actual header text to our known column names."""
    lower_headers = [h.lower() for h in header_texts]
    mapping: Dict[str, int] = {}
    for clean_name, patterns in COLUMN_PATTERNS.items():
        for pattern in patterns:
            for idx, header in enumerate(lower_headers):
                if pattern in header:
                    mapping[clean_name] = idx
                    break
            if clean_name in mapping:
                break
    return mapping


def _looks_like_gainers_table(df: pd.DataFrame) -> bool:
    """Heuristic check that a parsed table is actually the gainers table."""
    if df.empty or len(df.columns) < 4:
        return False
    col_text = " ".join(str(c).lower() for c in df.columns)
    return all(k in col_text for k in ("symbol", "price"))


def _parse_with_pandas_read_html(html: str) -> pd.DataFrame:
    """
    Primary parser: let pandas do the table/column alignment. This handles
    colspan/rowspan and any extra unlabeled cells (icons, star/watchlist
    buttons) correctly, which a manual <th>/<td> position-matching parser
    can silently get wrong.
    """
    try:
        tables = pd.read_html(html, flavor="html5lib")
    except Exception as e:
        logger.debug(f"pd.read_html failed: {e}")
        return pd.DataFrame()

    for raw_table in tables:
        if _looks_like_gainers_table(raw_table):
            column_index = _map_headers([str(c) for c in raw_table.columns])
            if "Symbol" not in column_index:
                continue
            renamed = {raw_table.columns[idx]: clean_name for clean_name, idx in column_index.items()}
            out = raw_table.rename(columns=renamed)
            keep_cols = [c for c in COLUMN_PATTERNS if c in out.columns]
            out = out[keep_cols].astype(str)
            return out.reindex(columns=list(COLUMN_PATTERNS.keys()))
    return pd.DataFrame()


def _parse_with_beautifulsoup(html: str) -> pd.DataFrame:
    """
    Fallback parser: manually walk <thead>/<tbody>. Handles the case where a
    data row has more <td> cells than the header has <th> cells (e.g. a
    leading star/watchlist icon column with no header label) by shifting
    the mapped indices by that row's cell-count surplus, rather than
    assuming header and data columns line up 1:1.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        logger.error("Could not find the gainers table on the page.")
        return pd.DataFrame()

    thead = table.find("thead")
    header_cells = thead.find_all("th") if thead else table.find_all("th")
    header_texts = [th.get_text(strip=True) for th in header_cells]
    if not header_texts:
        logger.error("Gainers table has no header row; cannot map columns.")
        return pd.DataFrame()

    column_index = _map_headers(header_texts)
    if "Symbol" not in column_index:
        logger.error(f"Could not locate a 'Symbol' column among headers: {header_texts}")
        return pd.DataFrame()

    logger.debug(f"Parsed headers: {header_texts}")
    logger.debug(f"Header -> column mapping: {column_index}")

    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]

    records: List[Dict[str, str]] = []
    for i, row in enumerate(rows):
        cells = row.find_all("td")
        if not cells:
            continue
        cell_texts = [c.get_text(strip=True) for c in cells]

        # If this row has more cells than the header does, assume the
        # surplus are unlabeled leading cells (icon/star/rank columns) and
        # shift every mapped index by that surplus.
        offset = max(0, len(cell_texts) - len(header_texts))

        record = {
            clean_name: cell_texts[idx + offset]
            for clean_name, idx in column_index.items()
            if (idx + offset) < len(cell_texts)
        }
        if i == 0:
            logger.debug(f"First raw row (offset={offset}): {record}")
        if record.get("Symbol"):
            records.append(record)

    return pd.DataFrame(records, columns=list(COLUMN_PATTERNS.keys()))


def fetch_gainers_data(url: str, headers: Dict[str, str]) -> pd.DataFrame:
    """
    Fetches and parses the top gainers table from Yahoo Finance.
    Returns a pandas DataFrame with raw (string) data, columns matching
    COLUMN_PATTERNS' keys where available.

    Tries pd.read_html first (handles column alignment automatically), then
    falls back to manual BeautifulSoup parsing if that doesn't find a
    plausible table.
    """
    session = _build_session()
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching gainers: {e}")
        return pd.DataFrame()

    try:
        df = _parse_with_pandas_read_html(response.text)
        if not df.empty:
            logger.info(f"Successfully fetched {len(df)} stocks from gainers list (pandas parser).")
            return df

        logger.info("pandas.read_html parse didn't yield a usable table; trying BeautifulSoup fallback.")
        df = _parse_with_beautifulsoup(response.text)
        logger.info(f"Successfully fetched {len(df)} stocks from gainers list (BeautifulSoup fallback).")
        return df

    except Exception as e:
        logger.error(f"Error parsing gainers data: {e}")
        return pd.DataFrame()


# --- Module 2: Data Validation & Enrichment ---
_SUFFIX_MULTIPLIERS = {"B": 1e9, "M": 1e6, "K": 1e3, "T": 1e12}


def _clean_currency(val: Any) -> float:
    """Parse strings like '$1.23B', '-4.56', '12,345' into floats."""
    if not isinstance(val, str):
        return val if isinstance(val, (int, float)) else np.nan
    text = val.replace("$", "").replace(",", "").strip()
    if not text or text.upper() in ("N/A", "NA", "-", "--"):
        return np.nan
    suffix = text[-1].upper() if text else ""
    multiplier = _SUFFIX_MULTIPLIERS.get(suffix)
    try:
        if multiplier:
            return float(text[:-1]) * multiplier
        return float(text)
    except ValueError:
        return np.nan


def _clean_percentage(val: Any) -> float:
    if not isinstance(val, str):
        return val if isinstance(val, (int, float)) else np.nan
    text = val.replace("%", "").replace(",", "").replace("+", "").strip()
    if not text or text.upper() in ("N/A", "NA", "-", "--"):
        return np.nan
    try:
        return float(text) / 100
    except ValueError:
        return np.nan


def _clean_volume(val: Any) -> float:
    """Returns a float (not int) since NaN can't be represented as int in pandas."""
    if not isinstance(val, str):
        return val if isinstance(val, (int, float)) else np.nan
    text = val.replace(",", "").strip()
    if not text or text.upper() in ("N/A", "NA", "-", "--"):
        return np.nan
    suffix = text[-1].upper() if text else ""
    multiplier = _SUFFIX_MULTIPLIERS.get(suffix)
    try:
        # Using float() first (not int()) matters: values like "1,234.5K" or
        # plain "1234.0" raise ValueError under int() but parse fine as float.
        if multiplier:
            return float(text[:-1]) * multiplier
        return float(text)
    except ValueError:
        return np.nan


_RANGE_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _split_52wk_range(range_text: Any) -> tuple[float, float]:
    """
    Parse a '52 Wk Range' string into (low, high).
    The page renders this as e.g. '12.34 - 56.78', so a naive split(' ')
    yields three tokens (['12.34', '-', '56.78']) instead of two — this
    pulls the two numeric tokens out directly instead of relying on a
    fixed separator/token count.
    """
    if not isinstance(range_text, str):
        return (np.nan, np.nan)
    numbers = _RANGE_NUMBER_RE.findall(range_text.replace(",", ""))
    if len(numbers) >= 2:
        try:
            return (float(numbers[0]), float(numbers[1]))
        except ValueError:
            return (np.nan, np.nan)
    return (np.nan, np.nan)


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw DataFrame: converts data types, handles missing values,
    and computes derived metrics. Always leaves the expected numeric
    columns present (as all-NaN if unparseable) so downstream code never
    hits a KeyError.
    """
    if df.empty:
        return df

    df = df.copy()

    df["Price"] = df.get("Price", pd.Series(np.nan, index=df.index)).apply(_clean_currency)
    df["Change"] = df.get("Change", pd.Series(np.nan, index=df.index)).apply(_clean_currency)
    df["Change %"] = df.get("Change %", pd.Series(np.nan, index=df.index)).apply(_clean_percentage)
    df["Volume"] = df.get("Volume", pd.Series(np.nan, index=df.index)).apply(_clean_volume)
    df["Avg Vol (3M)"] = df.get("Avg Vol (3M)", pd.Series(np.nan, index=df.index)).apply(_clean_volume)
    df["Market Cap"] = df.get("Market Cap", pd.Series(np.nan, index=df.index)).apply(_clean_currency)
    df["P/E Ratio (TTM)"] = pd.to_numeric(df.get("P/E Ratio (TTM)"), errors="coerce")
    df["52 Wk Change %"] = df.get("52 Wk Change %", pd.Series(np.nan, index=df.index)).apply(_clean_percentage)

    # Split 52 Wk Range into low/high — always creates both columns, even
    # when every value fails to parse, so later code can rely on them.
    low_high = df.get("52 Wk Range", pd.Series(np.nan, index=df.index)).apply(_split_52wk_range)
    df["52 Wk Low"] = low_high.apply(lambda t: t[0])
    df["52 Wk High"] = low_high.apply(lambda t: t[1])

    # Drop rows with critical missing data
    critical_cols = ["Symbol", "Price", "Volume", "Market Cap"]
    before = len(df)
    if before > 0:
        nan_counts = {c: int(df[c].isna().sum()) for c in critical_cols}
        if any(count == before for count in nan_counts.values()):
            # Every row lost a critical field — almost always a parsing
            # misalignment (wrong <td> mapped to a column) rather than
            # genuinely missing data. Log a raw sample so it's diagnosable
            # straight from the log instead of needing to reproduce.
            all_nan_cols = [c for c, count in nan_counts.items() if count == before]
            logger.error(
                f"Every row is missing {all_nan_cols} after cleaning — this usually means the "
                f"scraper mapped the wrong table cells to these columns (Yahoo's page layout may "
                f"have changed). Raw sample row: {df.iloc[0].to_dict()}"
            )
    df = df.dropna(subset=critical_cols).reset_index(drop=True)
    if before > 0 and df.empty:
        logger.error(
            f"All {before} rows dropped during cleaning. NaN counts per critical column: {nan_counts}"
        )

    # Calculate additional metrics, guarding against division by zero
    safe_market_cap = df["Market Cap"].replace(0, np.nan)
    safe_low = df["52 Wk Low"].replace(0, np.nan)
    safe_high = df["52 Wk High"].replace(0, np.nan)

    df["Volume/Market Cap"] = df["Volume"] / safe_market_cap  # Liquidity proxy
    df["Price/52Wk Low"] = df["Price"] / safe_low
    df["Price/52Wk High"] = df["Price"] / safe_high

    df = df.replace([np.inf, -np.inf], np.nan)

    logger.info(f"Data cleaned and validated. {len(df)} stocks remain.")
    return df


def _fetch_one_symbol_fundamentals(symbol: str) -> Dict[str, Any]:
    """Fetch fundamentals for a single symbol, with a couple of retries for
    transient errors (timeouts, rate limits)."""
    last_error: Optional[Exception] = None
    for attempt in range(1, FUNDAMENTALS_RETRY_ATTEMPTS + 1):
        try:
            info = yf.Ticker(symbol).info or {}
            return {
                "Symbol": symbol,
                "Beta": info.get("beta", np.nan),
                "Forward P/E": info.get("forwardPE", np.nan),
                "PEG Ratio": info.get("pegRatio", np.nan),
                "Profit Margin": info.get("profitMargins", np.nan),
                "Operating Margin": info.get("operatingMargins", np.nan),
                "ROE": info.get("returnOnEquity", np.nan),
                "Revenue Growth": info.get("revenueGrowth", np.nan),
                "EPS Growth": info.get("earningsGrowth", np.nan),
                "Debt/Equity": info.get("debtToEquity", np.nan),
                "Current Ratio": info.get("currentRatio", np.nan),
                "Dividend Yield": info.get("dividendYield", np.nan),
                "Short Ratio": info.get("shortRatio", np.nan),
            }
        except Exception as e:  # noqa: BLE001 - broad on purpose, we retry/log
            last_error = e
            if attempt < FUNDAMENTALS_RETRY_ATTEMPTS:
                time.sleep(FUNDAMENTALS_RETRY_DELAY_SECONDS * attempt)

    logger.warning(f"Could not fetch fundamentals for {symbol}: {last_error}")
    return {"Symbol": symbol}


def enrich_with_fundamentals(symbols: List[str]) -> pd.DataFrame:
    """
    Enriches stock data with additional fundamental metrics using yfinance.
    Fetched concurrently (bounded worker pool) since each call is a
    separate network round-trip — sequential fetching of even 20-30
    symbols can take tens of seconds.
    """
    if not symbols:
        return pd.DataFrame(columns=["Symbol"])

    logger.info(f"Fetching fundamental data for {len(symbols)} symbols...")
    results: Dict[str, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=min(MAX_FUNDAMENTALS_WORKERS, len(symbols))) as executor:
        future_to_symbol = {
            executor.submit(_fetch_one_symbol_fundamentals, symbol): symbol for symbol in symbols
        }
        completed = 0
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                results[symbol] = future.result()
            except Exception as e:  # defensive: _fetch_one_symbol_fundamentals already catches
                logger.warning(f"Unexpected failure fetching {symbol}: {e}")
                results[symbol] = {"Symbol": symbol}
            completed += 1
            if completed % 10 == 0 or completed == len(symbols):
                logger.info(f"Processed {completed}/{len(symbols)} symbols...")

    # Preserve the original symbol order rather than arbitrary completion order
    ordered = [results[s] for s in symbols if s in results]
    return pd.DataFrame(ordered)


# --- Module 3: Screening & Scoring Engine ---
def apply_screens(df: pd.DataFrame) -> pd.Series:
    """
    Applies a series of financial and technical screens to the DataFrame.
    Returns a boolean mask of passing stocks.
    """
    # 1. Basic Liquidity & Size Filter
    mask = (df["Price"] > 5) & (df["Volume"] > 100_000) & (df["Market Cap"] > 1e9)

    # 2. Momentum Filter - Avoid stocks that are extremely overextended from 52-week low/high.
    # 52 Wk Low/High may be NaN if the page didn't render a parseable range;
    # treat those as "unknown, don't screen out" rather than failing the mask.
    price_vs_low = df["Price/52Wk Low"]
    price_vs_high = df["Price/52Wk High"]
    momentum_ok = ((price_vs_low < 3) | price_vs_low.isna()) & ((price_vs_high > 0.7) | price_vs_high.isna())
    mask &= momentum_ok

    # 3. Volatility Filter (52-week range width relative to price)
    safe_price = df["Price"].replace(0, np.nan)
    range_width = (df["52 Wk High"] - df["52 Wk Low"]) / safe_price
    volatility_ok = (range_width < 0.5) | range_width.isna()
    mask &= volatility_ok

    logger.info(f"Applied basic screens. {int(mask.sum())} stocks passed.")
    return mask.fillna(False)


def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a composite score based on multiple factors to rank stocks.
    Higher score is better.
    """
    df = df.copy()

    # For P/E, lower is often better (sector-dependent, but a reasonable prior).
    df["PE_Score"] = 1 / df["P/E Ratio (TTM)"].clip(lower=1)

    # Robust z-scores (median/IQR) to reduce outlier influence
    z_scores = pd.DataFrame(index=df.index)
    for col in ["Change %", "Volume/Market Cap", "Price/52Wk Low", "PE_Score"]:
        if col not in df.columns:
            continue
        median = df[col].median()
        q75, q25 = df[col].quantile(0.75), df[col].quantile(0.25)
        iqr = q75 - q25
        z_scores[f"{col}_z"] = (df[col] - median) / iqr if iqr > 0 else 0.0

    weights = {
        "Change %_z": 0.4,
        "Volume/Market Cap_z": 0.2,
        "Price/52Wk Low_z": 0.2,  # Relative strength off the 52-week low
        "PE_Score_z": 0.2,
    }

    final_score = pd.Series(0.0, index=df.index)
    for col, weight in weights.items():
        if col in z_scores.columns:
            final_score += z_scores[col].fillna(0).clip(lower=-3, upper=3) * weight

    df["Composite_Score"] = final_score
    return df


# --- Main Orchestration Function ---
def stock_picking_pipeline() -> Optional[pd.DataFrame]:
    """
    The main pipeline that orchestrates data fetching, cleaning, enrichment,
    screening, scoring, and recommendation.
    """
    logger.info("Starting Stock Picking Pipeline...")

    # Step 1: Fetch raw data
    raw_df = fetch_gainers_data(GAINERS_URL, HEADERS)
    if raw_df.empty:
        logger.error("No data fetched. Exiting.")
        return None

    # Step 2: Clean and validate
    cleaned_df = clean_and_validate(raw_df)
    if cleaned_df.empty:
        logger.error("No data after cleaning. Exiting.")
        return None

    # Step 3: Apply initial screens
    screen_mask = apply_screens(cleaned_df)
    screened_df = cleaned_df[screen_mask].copy()
    if screened_df.empty:
        logger.warning("No stocks passed the initial screens. Expanding criteria...")
        screened_df = cleaned_df.copy()

    # Step 4: Enrich with fundamentals (only for passed stocks)
    symbols = screened_df["Symbol"].tolist()
    fundamental_df = enrich_with_fundamentals(symbols)

    # Merge with cleaned data
    enriched_df = pd.merge(screened_df, fundamental_df, on="Symbol", how="left")

    # Step 5: Apply fundamental screens.
    # Each metric independently "passes" if it's within a healthy range OR
    # unknown (NaN) — but a stock must pass ALL THREE metric checks, not
    # just any single one. (The original combined all six terms with a
    # single OR chain, which meant almost any stock passed as soon as one
    # metric was missing.)
    roe_ok = (enriched_df["ROE"] > 0.05) | enriched_df["ROE"].isna()
    debt_ok = (enriched_df["Debt/Equity"] < 1.5) | enriched_df["Debt/Equity"].isna()
    peg_ok = (enriched_df["PEG Ratio"] < 2) | enriched_df["PEG Ratio"].isna()
    fund_mask = roe_ok & debt_ok & peg_ok
    enriched_df = enriched_df[fund_mask].copy()

    if enriched_df.empty:
        logger.warning("No stocks passed the fundamental screens. Using pre-fundamental screen results.")
        enriched_df = pd.merge(screened_df, fundamental_df, on="Symbol", how="left")

    # Step 6: Compute composite score
    final_df = compute_composite_score(enriched_df)

    # Step 7: Rank and generate recommendations
    final_df = final_df.sort_values("Composite_Score", ascending=False)
    final_df["Recommendation"] = np.select(
        [
            final_df["Composite_Score"] > final_df["Composite_Score"].quantile(0.7),
            final_df["Composite_Score"] > final_df["Composite_Score"].quantile(0.3),
        ],
        ["Strong Buy", "Buy"],
        default="Hold",
    )

    # Step 8: Prepare output
    output_columns = [
        "Symbol", "Name", "Price", "Change %", "Composite_Score", "Recommendation",
        "P/E Ratio (TTM)", "ROE", "PEG Ratio", "Debt/Equity",
    ]
    output_columns = [col for col in output_columns if col in final_df.columns]

    result_df = final_df[output_columns].head(10)  # Top 10 picks
    logger.info("Pipeline completed successfully.")
    return result_df


# --- Execution & Output ---
if __name__ == "__main__":
    result = stock_picking_pipeline()
    if result is not None and not result.empty:
        print("\n" + "=" * 80)
        print("TOP STOCK PICKS BASED ON TODAY'S GAINERS")
        print("=" * 80)
        print(result.to_string(index=False))
        print("\nNote: This is a quantitative screen and not financial advice.")
    else:
        print("Pipeline failed to produce results.")
