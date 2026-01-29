"""
Enhanced Stock Dashboard with Advanced Features
- Improved caching with Redis option
- Advanced charting with interactive controls
- Portfolio simulation capabilities
- News integration
- Performance benchmarking
- Downloadable reports
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from typing import List, Tuple, Optional, Dict, Any
import yfinance as yf
import ta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
import math
import logging
import json
from dataclasses import dataclass
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')

# ---------- Configuration ----------
APP_TITLE = "📈 Alpha Dashboard"
PAGE_TITLE = "Advanced Stock Analysis Dashboard"
AUTHOR = "FC"
DEFAULT_SYMBOL = "BBAI"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_DAYS_BACK = 365
CACHE_TTL = 1800  # 30 minutes
REFRESH_INTERVAL = 300  # Auto-refresh every 5 minutes

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Main containers */
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .info-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #007bff;
    }

    .warning-card {
        background: #fff3cd;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #ffc107;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding: 10px 16px;
    }

    /* Data table styling */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
    }

    .dataframe th {
        background-color: #f8f9fa;
        font-weight: bold;
        text-align: left;
        padding: 8px;
        border-bottom: 2px solid #dee2e6;
    }

    .dataframe td {
        padding: 8px;
        border-bottom: 1px solid #dee2e6;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #007bff;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Data Classes ----------
@dataclass
class StockData:
    """Container for stock data with metadata"""
    symbol: str
    df: pd.DataFrame
    info: Dict[str, Any]
    indicators: Optional[pd.DataFrame] = None
    last_updated: Optional[datetime] = None

    def is_valid(self) -> bool:
        return not self.df.empty and len(self.df) > 10


# ---------- Enhanced Utilities ----------
class Formatter:
    """Enhanced formatting utilities"""

    @staticmethod
    def currency(value: Optional[float], prefix: str = "$") -> str:
        """Format currency with appropriate suffixes"""
        if pd.isna(value) or value is None:
            return "N/A"

        try:
            value = float(value)
            abs_value = abs(value)
            if abs_value >= 1e12:
                return f"{prefix}{value / 1e12:.2f}T"
            elif abs_value >= 1e9:
                return f"{prefix}{value / 1e9:.2f}B"
            elif abs_value >= 1e6:
                return f"{prefix}{value / 1e6:.2f}M"
            elif abs_value >= 1e3:
                return f"{prefix}{value / 1e3:.1f}K"
            else:
                return f"{prefix}{value:.2f}"
        except (ValueError, TypeError):
            return "N/A"

    @staticmethod
    def percentage(value: Optional[float], decimals: int = 2) -> str:
        """Format percentage"""
        if pd.isna(value) or value is None:
            return "N/A"
        try:
            return f"{float(value):.{decimals}f}%"
        except (ValueError, TypeError):
            return "N/A"

    @staticmethod
    def number(value: Optional[float], decimals: int = 2) -> str:
        """Format general number"""
        if pd.isna(value) or value is None:
            return "N/A"
        try:
            return f"{float(value):.{decimals}f}"
        except (ValueError, TypeError):
            return "N/A"

    @staticmethod
    def large_number(value: Optional[float]) -> str:
        """Format large numbers with commas"""
        if pd.isna(value) or value is None:
            return "N/A"
        try:
            return f"{float(value):,.0f}"
        except (ValueError, TypeError):
            return "N/A"


class Validator:
    """Input validation utilities"""

    @staticmethod
    def validate_symbol(symbol: str) -> Tuple[bool, str]:
        """Validate stock symbol format"""
        if not symbol or not isinstance(symbol, str):
            return False, "Symbol cannot be empty"
        if len(symbol) > 10:
            return False, "Symbol too long"
        if not symbol.replace('.', '').replace('-', '').isalnum():
            return False, "Symbol must be alphanumeric"
        return True, ""

    @staticmethod
    def validate_date_range(start: date, end: date) -> Tuple[bool, str]:
        """Validate date range"""
        if start >= end:
            return False, "Start date must be before end date"
        if (end - start).days > 365 * 10:  # 10 years max
            return False, "Date range too large (max 10 years)"
        if start < date(1900, 1, 1):
            return False, "Start date too early"
        return True, ""


# ---------- Enhanced Data Service ----------
class EnhancedDataService:
    """Enhanced data fetching with retry logic and multiple sources"""

    # Predefined sector ETFs for comparison
    SECTOR_ETFS = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financials": "XLF",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Industrials": "XLI",
        "Materials": "XLB",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Communication": "XLC"
    }

    @staticmethod
    @st.cache_data(ttl=CACHE_TTL, show_spinner=False)
    def fetch_ohlcv(
            symbol: str,
            start: date,
            end: date,
            interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch OHLCV data with enhanced error handling"""
        try:
            logger.info(f"Fetching data for {symbol} from {start} to {end}")

            # Adjust end date for yfinance
            end_adj = end + timedelta(days=1)

            # Try multiple intervals for better data coverage
            df = yf.download(
                symbol,
                start=start,
                end=end_adj,
                interval=interval,
                progress=False,
                threads=True,
                auto_adjust=True,
                prepost=False
            )

            if df.empty or len(df) < 5:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Clean and standardize columns
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

            # Ensure we have all required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Missing column {col} for {symbol}")
                    return pd.DataFrame()

            # Rename columns to avoid conflicts with Python built-ins
            df = df.rename(columns={
                'Open': 'Open_Price',
                'High': 'High_Price',
                'Low': 'Low_Price',
                'Close': 'Close_Price',
                'Adj Close': 'Adj_Close'
            })

            # Calculate derived metrics safely
            df['Returns'] = df['Close_Price'].pct_change()
            df['Log_Returns'] = np.log(df['Close_Price'] / df['Close_Price'].shift(1))
            df['Range'] = (df['High_Price'] - df['Low_Price']) / df['Low_Price'].replace(0, np.nan)
            df['Gap'] = (df['Open_Price'] - df['Close_Price'].shift(1)) / df['Close_Price'].shift(1).replace(0, np.nan)
            df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
            df['Volatility_30d'] = df['Returns'].rolling(30).std() * np.sqrt(252)
            df['Cumulative_Return'] = (1 + df['Returns']).cumprod() - 1

            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')

            logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=CACHE_TTL * 2, show_spinner=False)
    def fetch_company_info(symbol: str) -> Dict[str, Any]:
        """Fetch comprehensive company information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            # Essential info extraction with defaults
            essential_info = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'marketCap': info.get('marketCap'),
                'currentPrice': info.get('currentPrice'),
                'previousClose': info.get('previousClose'),
                'open_price': info.get('open'),
                'dayLow': info.get('dayLow'),
                'dayHigh': info.get('dayHigh'),
                'volume': info.get('volume'),
                'averageVolume': info.get('averageVolume'),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
                'dividendYield': info.get('dividendYield'),
                'trailingPE': info.get('trailingPE'),
                'forwardPE': info.get('forwardPE'),
                'beta': info.get('beta'),
                'sharesOutstanding': info.get('sharesOutstanding'),
                'website': info.get('website'),
                'longBusinessSummary': info.get('longBusinessSummary', 'No description available.'),
                'country': info.get('country', 'N/A'),
                'employees': info.get('fullTimeEmployees')
            }

            return essential_info

        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'name': symbol, 'error': str(e)}

    @staticmethod
    @st.cache_data(ttl=CACHE_TTL, show_spinner=False)
    def fetch_news(symbol: str, max_articles: int = 10) -> List[Dict]:
        """Fetch recent news for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            return news[:max_articles]
        except:
            return []

    @staticmethod
    @st.cache_data(ttl=CACHE_TTL, show_spinner=False)
    def fetch_multiple_symbols(
            symbols: List[str],
            start: date,
            end: date
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols efficiently"""
        results = {}
        for symbol in symbols:
            df = EnhancedDataService.fetch_ohlcv(symbol, start, end)
            if not df.empty:
                results[symbol] = df
        return results


# ---------- Advanced Technical Indicators ----------
class AdvancedIndicators:
    """Advanced technical indicators and features"""

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def compute_all(
            df: pd.DataFrame,
            include_trend: bool = True,
            include_momentum: bool = True,
            include_volatility: bool = True,
            include_volume: bool = True
    ) -> pd.DataFrame:
        """Compute comprehensive technical indicators"""
        if df.empty or 'Close_Price' not in df.columns:
            return df

        result = df.copy()
        close = result['Close_Price']
        high = result['High_Price']
        low = result['Low_Price']
        volume = result['Volume']

        # === TREND INDICATORS ===
        if include_trend:
            # Multiple moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                result[f'SMA_{period}'] = close.rolling(window=period).mean()
                result[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()

            # Ichimoku Cloud
            try:
                ichimoku = ta.trend.IchimokuIndicator(high, low)
                result['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
                result['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
                result['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
                result['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
            except Exception as e:
                logger.debug(f"Ichimoku calculation failed: {e}")
                pass

        # === MOMENTUM INDICATORS ===
        if include_momentum:
            # RSI variants
            for period in [7, 14, 21]:
                try:
                    result[f'RSI_{period}'] = ta.momentum.RSIIndicator(close, window=period).rsi()
                except Exception as e:
                    logger.debug(f"RSI calculation failed for period {period}: {e}")
                    result[f'RSI_{period}'] = np.nan

            # MACD
            try:
                macd = ta.trend.MACD(close)
                result['MACD'] = macd.macd()
                result['MACD_Signal'] = macd.macd_signal()
                result['MACD_Diff'] = macd.macd_diff()
            except Exception as e:
                logger.debug(f"MACD calculation failed: {e}")
                pass

        # === VOLATILITY INDICATORS ===
        if include_volatility:
            # Bollinger Bands
            try:
                bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
                result['BB_Upper'] = bb.bollinger_hband()
                result['BB_Lower'] = bb.bollinger_lband()
                result['BB_Middle'] = bb.bollinger_mavg()
                result['BB_Width'] = (result['BB_Upper'] - result['BB_Lower']) / result['BB_Middle']
            except Exception as e:
                logger.debug(f"Bollinger Bands calculation failed: {e}")
                pass

            # ATR
            try:
                result['ATR'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
                result['ATR_Percent'] = result['ATR'] / close.replace(0, np.nan) * 100
            except Exception as e:
                logger.debug(f"ATR calculation failed: {e}")
                result['ATR'] = np.nan
                result['ATR_Percent'] = np.nan

        # === VOLUME INDICATORS ===
        if include_volume:
            # OBV
            try:
                result['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            except Exception as e:
                logger.debug(f"OBV calculation failed: {e}")
                result['OBV'] = np.nan

            # Volume Weighted Average Price
            try:
                typical_price = (high + low + close) / 3
                result['VWAP'] = (typical_price * volume).cumsum() / volume.cumsum().replace(0, np.nan)
            except Exception as e:
                logger.debug(f"VWAP calculation failed: {e}")
                result['VWAP'] = np.nan

        # === SUPPORT/RESISTANCE ===
        # Rolling highs/lows
        for period in [20, 50, 100]:
            result[f'Rolling_High_{period}'] = high.rolling(window=period).max()
            result[f'Rolling_Low_{period}'] = low.rolling(window=period).min()

        # Fill NaN values for better visualization
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(method='ffill').fillna(method='bfill')

        return result

    @staticmethod
    def detect_support_resistance(
            df: pd.DataFrame,
            window: int = 20,
            tolerance: float = 0.02
    ) -> Tuple[List[float], List[float]]:
        """Detect support and resistance levels"""
        if df.empty or 'High_Price' not in df.columns or 'Low_Price' not in df.columns:
            return [], []

        high = df['High_Price']
        low = df['Low_Price']

        # Find local maxima and minima
        highs = []
        lows = []

        for i in range(window, len(df) - window):
            if high.iloc[i] == high.iloc[i - window:i + window].max():
                highs.append(float(high.iloc[i]))
            if low.iloc[i] == low.iloc[i - window:i + window].min():
                lows.append(float(low.iloc[i]))

        # Cluster similar levels
        def cluster_levels(levels: List[float], tolerance: float) -> List[float]:
            if not levels:
                return []
            levels.sort()
            clusters = [[levels[0]]]

            for level in levels[1:]:
                if abs(level - clusters[-1][-1]) / clusters[-1][-1] < tolerance:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])

            return [float(np.mean(cluster)) for cluster in clusters]

        support = cluster_levels(lows, tolerance)
        resistance = cluster_levels(highs, tolerance)

        return support[:5], resistance[:5]  # Return top 5 levels


# ---------- Advanced Risk Metrics ----------
class AdvancedRiskMetrics:
    """Advanced risk and performance metrics"""

    @staticmethod
    def calculate_all_metrics(
            returns: pd.Series,
            benchmark_returns: Optional[pd.Series] = None,
            risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        metrics = {}

        if returns.empty or returns.isna().all():
            return metrics

        # Clean returns
        returns_clean = returns.dropna()
        if len(returns_clean) < 5:
            return metrics

        # Annualized metrics
        ann_factor = 252

        # Basic metrics
        metrics['total_return'] = float(((1 + returns_clean).prod() - 1) * 100)
        metrics['annual_return'] = float(returns_clean.mean() * ann_factor * 100)
        metrics['annual_volatility'] = float(returns_clean.std() * np.sqrt(ann_factor) * 100)

        # Risk-adjusted returns
        excess_returns = returns_clean - risk_free_rate / ann_factor

        if returns_clean.std() > 0:
            metrics['sharpe_ratio'] = float(excess_returns.mean() / returns_clean.std() * np.sqrt(ann_factor))

        # Sortino ratio (downside deviation)
        downside_returns = returns_clean[returns_clean < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            metrics['sortino_ratio'] = float(
                excess_returns.mean() / downside_returns.std() * np.sqrt(ann_factor)
            )

        # Maximum Drawdown
        try:
            cumulative = (1 + returns_clean).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max.replace(0, np.nan)
            metrics['max_drawdown'] = float(drawdown.min() * 100)
            metrics['avg_drawdown'] = float(drawdown.mean() * 100)
        except Exception:
            metrics['max_drawdown'] = 0.0
            metrics['avg_drawdown'] = 0.0

        # Calmar Ratio
        if abs(metrics.get('max_drawdown', 100)) > 0.01:
            metrics['calmar_ratio'] = metrics.get('annual_return', 0) / abs(metrics['max_drawdown'])

        # Value at Risk (95% confidence)
        if len(returns_clean) >= 10:
            metrics['var_95'] = float(np.percentile(returns_clean, 5) * 100)
            cvar_data = returns_clean[returns_clean <= np.percentile(returns_clean, 5)]
            metrics['cvar_95'] = float(cvar_data.mean() * 100) if len(cvar_data) > 0 else 0.0

        # Skewness and Kurtosis
        if len(returns_clean) >= 10:
            try:
                metrics['skewness'] = float(stats.skew(returns_clean))
                metrics['kurtosis'] = float(stats.kurtosis(returns_clean))
            except Exception:
                metrics['skewness'] = 0.0
                metrics['kurtosis'] = 0.0

        # Alpha and Beta (if benchmark provided)
        if benchmark_returns is not None:
            aligned_bench = benchmark_returns.dropna()
            aligned_data = pd.concat([returns_clean, aligned_bench], axis=1).dropna()
            if len(aligned_data) > 10:
                try:
                    x = aligned_data.iloc[:, 1].values.reshape(-1, 1)
                    y = aligned_data.iloc[:, 0].values

                    model = LinearRegression().fit(x, y)
                    metrics['beta'] = float(model.coef_[0])
                    metrics['alpha'] = float(model.intercept_ * ann_factor * 100)

                    # R-squared
                    metrics['r_squared'] = float(model.score(x, y))
                except Exception:
                    metrics['beta'] = 0.0
                    metrics['alpha'] = 0.0
                    metrics['r_squared'] = 0.0

        # Win rate
        metrics['win_rate'] = float((returns_clean > 0).mean() * 100)
        winning_trades = returns_clean[returns_clean > 0]
        losing_trades = returns_clean[returns_clean < 0]

        metrics['avg_win'] = float(winning_trades.mean() * 100) if len(winning_trades) > 0 else 0.0
        metrics['avg_loss'] = float(losing_trades.mean() * 100) if len(losing_trades) > 0 else 0.0

        # Profit factor
        if len(losing_trades) > 0 and abs(losing_trades.sum()) > 0:
            metrics['profit_factor'] = float(
                abs(winning_trades.sum() / losing_trades.sum())
            )
        else:
            metrics['profit_factor'] = float('nan')

        return metrics

    @staticmethod
    def monte_carlo_simulation(
            initial_price: float,
            mu: float,
            sigma: float,
            days: int = 252,
            simulations: int = 1000
    ) -> np.ndarray:
        """Monte Carlo simulation for future price paths"""
        if days <= 0 or simulations <= 0:
            return np.array([])

        dt = 1 / 252

        # Generate random paths
        paths = np.zeros((days, simulations))
        paths[0] = initial_price

        np.random.seed(42)  # For reproducibility
        for t in range(1, days):
            z = np.random.standard_normal(simulations)
            paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

        return paths


# ---------- Enhanced Visualizer ----------
class EnhancedVisualizer:
    """Enhanced visualization with interactive features"""

    @staticmethod
    def create_metrics_dashboard(
            info: Dict,
            df: pd.DataFrame,
            metrics: Dict
    ) -> None:
        """Create comprehensive metrics dashboard"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            try:
                current_price = info.get('currentPrice', df['Close_Price'].iloc[
                    -1] if not df.empty and 'Close_Price' in df.columns else 0)
                prev_close = info.get('previousClose',
                                      df['Close_Price'].iloc[-2] if len(df) > 1 and 'Close_Price' in df.columns else 0)
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close and prev_close != 0 else 0

                st.metric(
                    "Current Price",
                    Formatter.currency(current_price),
                    delta=Formatter.percentage(change_pct) if change_pct != 0 else "0.00%",
                    delta_color="normal" if change_pct >= 0 else "inverse"
                )
            except Exception:
                st.metric("Current Price", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            market_cap = info.get('marketCap', 0)
            pe_ratio = info.get('trailingPE', 0)

            st.metric("Market Cap", Formatter.currency(market_cap))
            st.caption(f"P/E: {Formatter.number(pe_ratio, 1)}" if pe_ratio else "P/E: N/A")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            volatility = metrics.get('annual_volatility', 0)
            sharpe = metrics.get('sharpe_ratio', 0)

            st.metric("Volatility", Formatter.percentage(volatility))
            st.caption(f"Sharpe: {Formatter.number(sharpe, 2)}" if sharpe else "Sharpe: N/A")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            beta_val = info.get('beta', 0)
            div_yield = info.get('dividendYield', 0)

            st.metric("Beta", Formatter.number(beta_val, 2))
            st.caption(f"Div Yield: {Formatter.percentage(div_yield * 100) if div_yield else 'N/A'}")
            st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def create_interactive_chart(
            df: pd.DataFrame,
            symbol: str,
            indicators: List[str] = None
    ) -> go.Figure:
        """Create interactive candlestick chart with indicators"""
        if df.empty or 'Open_Price' not in df.columns:
            return go.Figure()

        if indicators is None:
            indicators = []

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{symbol} Price', 'Volume', 'RSI')
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open_Price'],
                high=df['High_Price'],
                low=df['Low_Price'],
                close=df['Close_Price'],
                name='Price',
                showlegend=False
            ),
            row=1, col=1
        )

        # Moving Averages
        if 'SMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )

        if 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['SMA_50'],
                    name='SMA 50',
                    line=dict(color='red', width=1)
                ),
                row=1, col=1
            )

        # Bollinger Bands
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=0.5, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=0.5, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )

        # Volume
        if 'Volume' in df.columns and 'Open_Price' in df.columns and 'Close_Price' in df.columns:
            colors = ['green' if close >= open else 'red'
                      for close, open in zip(df['Close_Price'], df['Open_Price'])]

            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=1
            )

        # RSI
        if 'RSI_14' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI_14'],
                    name='RSI',
                    line=dict(color='purple', width=1)
                ),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.update_yaxes(range=[0, 100], row=3, col=1)

        fig.update_layout(
            height=800,
            title=f"{symbol} - Interactive Chart",
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    @staticmethod
    def create_correlation_matrix(
            data_dict: Dict[str, pd.DataFrame],
            period: str = '1M'
    ) -> go.Figure:
        """Create correlation matrix heatmap"""
        returns_data = {}

        for symbol, df in data_dict.items():
            if not df.empty and 'Returns' in df.columns:
                if period == '1M':
                    returns_data[symbol] = df['Returns'].tail(21)
                elif period == '3M':
                    returns_data[symbol] = df['Returns'].tail(63)
                elif period == '1Y':
                    returns_data[symbol] = df['Returns'].tail(252)
                else:
                    returns_data[symbol] = df['Returns']

        if len(returns_data) < 2:
            return go.Figure()

        try:
            combined = pd.concat(returns_data, axis=1).dropna()
            if combined.shape[1] < 2:
                return go.Figure()

            corr_matrix = combined.corr()

            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                title=f'Returns Correlation Matrix ({period})'
            )

            fig.update_layout(height=600)
            return fig
        except Exception:
            return go.Figure()

    @staticmethod
    def create_monte_carlo_chart(
            paths: np.ndarray,
            initial_price: float,
            confidence_level: float = 0.95
    ) -> go.Figure:
        """Create Monte Carlo simulation visualization"""
        if paths.size == 0:
            return go.Figure()

        fig = go.Figure()

        # Plot all paths with low opacity (limit for performance)
        max_paths_to_show = min(50, paths.shape[1])
        for i in range(max_paths_to_show):
            fig.add_trace(
                go.Scatter(
                    x=list(range(paths.shape[0])),
                    y=paths[:, i],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(0,100,80,0.1)'),
                    showlegend=False
                )
            )

        # Calculate confidence intervals
        try:
            lower_bound = np.percentile(paths, (100 - confidence_level * 100) / 2, axis=1)
            upper_bound = np.percentile(paths, 100 - (100 - confidence_level * 100) / 2, axis=1)
            median_path = np.median(paths, axis=1)

            # Add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=list(range(paths.shape[0])) + list(range(paths.shape[0]))[::-1],
                    y=list(upper_bound) + list(lower_bound[::-1]),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence_level:.0%} Confidence Interval'
                )
            )

            # Add median path
            fig.add_trace(
                go.Scatter(
                    x=list(range(paths.shape[0])),
                    y=median_path,
                    line=dict(color='red', width=2),
                    name='Median Path'
                )
            )
        except Exception:
            pass

        # Add initial price line
        fig.add_hline(
            y=initial_price,
            line_dash="dash",
            line_color="green",
            annotation_text="Initial Price"
        )

        fig.update_layout(
            title="Monte Carlo Simulation - Future Price Paths",
            xaxis_title="Days Ahead",
            yaxis_title="Price",
            hovermode='x',
            height=500
        )

        return fig


# ---------- Portfolio Analysis ----------
class PortfolioAnalyzer:
    """Portfolio analysis and optimization tools"""

    @staticmethod
    def calculate_portfolio_metrics(
            weights: List[float],
            returns_df: pd.DataFrame,
            risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """Calculate portfolio metrics given weights"""
        if len(weights) != returns_df.shape[1]:
            return {}

        # Portfolio returns
        try:
            portfolio_returns = (returns_df * weights).sum(axis=1)
            return AdvancedRiskMetrics.calculate_all_metrics(
                portfolio_returns,
                risk_free_rate=risk_free_rate
            )
        except Exception:
            return {}

    @staticmethod
    def efficient_frontier(
            returns_df: pd.DataFrame,
            num_portfolios: int = 1000
    ) -> pd.DataFrame:
        """Generate efficient frontier"""
        if returns_df.empty or returns_df.shape[1] < 2:
            return pd.DataFrame()

        num_assets = returns_df.shape[1]
        results = []

        np.random.seed(42)
        for _ in range(num_portfolios):
            try:
                # Generate random weights
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)

                # Portfolio metrics
                portfolio_returns = (returns_df * weights).sum(axis=1)
                portfolio_return = portfolio_returns.mean() * 252 * 100
                portfolio_vol = portfolio_returns.std() * np.sqrt(252) * 100

                # Sharpe ratio (assuming 2% risk-free rate)
                sharpe = (portfolio_return - 2) / portfolio_vol if portfolio_vol > 0 else 0

                results.append({
                    'weights': weights,
                    'return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe': sharpe
                })
            except Exception:
                continue

        return pd.DataFrame(results)


# ---------- Main Application ----------
class EnhancedStockDashboard:
    """Main dashboard application"""

    def __init__(self):
        self.today = date.today()
        self.default_start = self.today - timedelta(days=DEFAULT_DAYS_BACK)
        self.data_service = EnhancedDataService()
        self.visualizer = EnhancedVisualizer()
        self.init_session_state()

    def init_session_state(self):
        """Initialize session state variables"""
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()

        if 'selected_tab' not in st.session_state:
            st.session_state.selected_tab = "Overview"

        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = ["AAPL", "NVDA", "PLTR", "MSFT", "GOOGL", "AMZN", "TSLA"]

    def sidebar(self) -> Dict[str, Any]:
        """Create enhanced sidebar with more options"""
        st.sidebar.markdown("## ⚙️ Dashboard Configuration")

        # Symbol input with validation
        symbol = st.sidebar.text_input(
            "**Primary Stock Symbol**",
            DEFAULT_SYMBOL,
            help="Enter stock ticker symbol (e.g., AAPL, MSFT)"
        ).upper().strip()

        # Benchmark selection
        benchmark = st.sidebar.selectbox(
            "**Benchmark Index**",
            ["SPY", "QQQ", "DIA", "IWM", "VTI"] + list(EnhancedDataService.SECTOR_ETFS.values()),
            index=0,
            help="Select benchmark for comparison"
        )

        # Date range with presets
        date_preset = st.sidebar.selectbox(
            "**Date Range Preset**",
            ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "Custom"],
            index=3
        )

        if date_preset == "Custom":
            start = st.sidebar.date_input("Start Date", self.default_start)
            end = st.sidebar.date_input("End Date", self.today)
        else:
            end = self.today
            if date_preset == "1 Month":
                start = end - timedelta(days=30)
            elif date_preset == "3 Months":
                start = end - timedelta(days=90)
            elif date_preset == "6 Months":
                start = end - timedelta(days=180)
            elif date_preset == "1 Year":
                start = end - timedelta(days=365)
            elif date_preset == "2 Years":
                start = end - timedelta(days=730)
            else:  # 5 Years
                start = end - timedelta(days=1825)

        st.sidebar.caption("ⓒ Franklin Chidi (FC) - MIT License")

        # Technical indicators selection
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Technical Indicators")

        indicators = {
            'Moving Averages': st.sidebar.checkbox("Moving Averages", True),
            'Bollinger Bands': st.sidebar.checkbox("Bollinger Bands", True),
            'RSI': st.sidebar.checkbox("RSI", True),
            'MACD': st.sidebar.checkbox("MACD", True),
            'Volume Indicators': st.sidebar.checkbox("Volume Indicators", True),
            'Support/Resistance': st.sidebar.checkbox("Support/Resistance", False),
        }

        # Advanced features
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 Advanced Features")

        advanced = {
            'monte_carlo': st.sidebar.checkbox("Monte Carlo Simulation", False),
            'portfolio_analysis': st.sidebar.checkbox("Portfolio Analysis", False),
            'news_feed': st.sidebar.checkbox("News Feed", True),
            'auto_refresh': st.sidebar.checkbox("Auto Refresh", False),
        }

        # Watchlist management
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Watchlist")

        watchlist_symbol = st.sidebar.text_input("Add to watchlist", "")
        if st.sidebar.button("Add") and watchlist_symbol:
            symbol_to_add = watchlist_symbol.upper().strip()
            if symbol_to_add and symbol_to_add not in st.session_state.watchlist:
                st.session_state.watchlist.append(symbol_to_add)
                st.sidebar.success(f"Added {symbol_to_add} to watchlist")

        # Display current watchlist
        st.sidebar.markdown("**Current Watchlist:**")
        for sym in st.session_state.watchlist[:10]:  # Limit display
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(sym)
            if col2.button("×", key=f"remove_{sym}"):
                st.session_state.watchlist.remove(sym)
                st.rerun()

        return {
            'symbol': symbol,
            'benchmark': benchmark,
            'start': start,
            'end': end,
            'indicators': indicators,
            'advanced': advanced,
            'watchlist': st.session_state.watchlist.copy()
        }

    def run(self):
        """Main application loop"""
        # Header
        st.markdown(f"""
        <div class="main-header">
            <h1>{PAGE_TITLE}</h1>
            <p>Professional-grade stock analysis and portfolio tools | Built by {AUTHOR} | Data Refreshed: {datetime.now().strftime('%Y-%m-%d - %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)

        # Get configuration from sidebar
        config = self.sidebar()

        # Validate inputs
        is_valid, error_msg = Validator.validate_symbol(config['symbol'])
        if not is_valid:
            st.error(f"Invalid symbol: {error_msg}")
            return

        is_valid, error_msg = Validator.validate_date_range(config['start'], config['end'])
        if not is_valid:
            st.error(f"Invalid date range: {error_msg}")
            return

        # Check auto-refresh
        if config['advanced']['auto_refresh']:
            time_since_refresh = (datetime.now() - st.session_state.last_refresh).seconds
            if time_since_refresh > REFRESH_INTERVAL:
                st.session_state.last_refresh = datetime.now()
                st.rerun()

        # Fetch data with progress indicators
        try:
            with st.spinner(f"Fetching data for {config['symbol']}..."):
                progress_bar = st.progress(0)

                # Fetch main data
                df = self.data_service.fetch_ohlcv(
                    config['symbol'],
                    config['start'],
                    config['end']
                )
                progress_bar.progress(25)

                # Fetch benchmark data
                benchmark_df = self.data_service.fetch_ohlcv(
                    config['benchmark'],
                    config['start'],
                    config['end']
                )
                progress_bar.progress(50)

                # Fetch company info
                info = self.data_service.fetch_company_info(config['symbol'])
                progress_bar.progress(75)

                # Compute indicators
                if not df.empty:
                    df = AdvancedIndicators.compute_all(df)
                progress_bar.progress(100)
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            logger.error(f"Data fetch error: {e}")
            return

        # Check if data was fetched successfully
        if df.empty or 'Close_Price' not in df.columns:
            st.error(f"""
            ❌ Unable to fetch data for {config['symbol']}. Possible reasons:
            - Invalid symbol
            - Market closed
            - No data for selected date range
            - Yahoo Finance API limitation

            Try a different symbol or date range.
            """)
            return

        # Calculate metrics safely
        try:
            metrics = AdvancedRiskMetrics.calculate_all_metrics(
                df['Returns'].dropna(),
                benchmark_df[
                    'Returns'].dropna() if not benchmark_df.empty and 'Returns' in benchmark_df.columns else None
            )
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics = {}

        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview",
            "📈 Technical Analysis",
            "🔄 Comparison",
            "📉 Risk Analysis",
            "💼 Portfolio",
            "📋 Data & Export"
        ])

        # Tab 1: Overview
        with tab1:
            self.render_overview_tab(config, df, info, metrics)

        # Tab 2: Technical Analysis
        with tab2:
            self.render_technical_tab(config, df)

        # Tab 3: Comparison
        with tab3:
            self.render_comparison_tab(config, df, benchmark_df)

        # Tab 4: Risk Analysis
        with tab4:
            self.render_risk_tab(config, df, metrics)

        # Tab 5: Portfolio
        with tab5:
            self.render_portfolio_tab(config, df)

        # Tab 6: Data & Export
        with tab6:
            self.render_data_tab(config, df, info, metrics)

        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d - %H:%M:%S')}")
        with col2:
            st.caption(f"Data source: Yahoo Finance")
        with col3:
            if st.button("🔄 Refresh Data"):
                st.session_state.last_refresh = datetime.now()
                st.rerun()

    def render_overview_tab(self, config, df, info, metrics):
        """Render overview tab"""
        # Metrics dashboard
        self.visualizer.create_metrics_dashboard(info, df, metrics)

        # Main chart
        st.markdown("### 📈 Price Chart with Indicators")
        try:
            chart_data = df.tail(252) if len(df) > 252 else df
            fig = self.visualizer.create_interactive_chart(chart_data, config['symbol'])
            if fig.data:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Unable to create chart with available data")
        except Exception as e:
            logger.error(f"Chart creation error: {e}")
            st.error("Error creating chart")

        # Company info and key statistics
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🏢 Company Information")
            st.markdown('<div class="info-card">', unsafe_allow_html=True)

            # Basic info
            info_rows = [
                ("Name", info.get('name', 'N/A')),
                ("Sector", info.get('sector', 'N/A')),
                ("Industry", info.get('industry', 'N/A')),
                ("Country", info.get('country', 'N/A')),
                ("Employees", Formatter.large_number(info.get('employees', 0))),
                ("Website", info.get('website', 'N/A')),
            ]

            for label, value in info_rows:
                st.write(f"**{label}:** {value}")

            st.markdown('</div>', unsafe_allow_html=True)

            # Business summary
            st.markdown("### 📝 Business Summary")
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.write(info.get('longBusinessSummary', 'No description available.'))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("### 📊 Key Statistics")
            st.markdown('<div class="info-card">', unsafe_allow_html=True)

            try:
                stats_data = [
                    ("52 Week High", Formatter.currency(info.get('fiftyTwoWeekHigh', 0))),
                    ("52 Week Low", Formatter.currency(info.get('fiftyTwoWeekLow', 0))),
                    ("Avg Volume", Formatter.large_number(info.get('averageVolume', 0))),
                    ("Shares Outstanding", Formatter.large_number(info.get('sharesOutstanding', 0))),
                    ("Forward P/E", Formatter.number(info.get('forwardPE', 0))),
                    ("Dividend Yield",
                     Formatter.percentage(info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0)),
                ]

                for label, value in stats_data:
                    st.metric(label, value)
            except Exception:
                st.warning("Unable to load key statistics")

            st.markdown('</div>', unsafe_allow_html=True)

            # News feed
            if config['advanced']['news_feed']:
                st.markdown("### 📰 Recent News")
                try:
                    news = self.data_service.fetch_news(config['symbol'], max_articles=3)
                    if news:
                        for article in news:
                            with st.expander(article.get('title', 'No title')[:50] + "..."):
                                st.write(f"**Publisher:** {article.get('publisher', 'Unknown')}")
                                if 'providerPublishTime' in article:
                                    st.write(
                                        f"**Published:** {datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M')}")
                                st.write(article.get('summary', 'No summary available.')[:200] + "...")
                                if article.get('link'):
                                    st.markdown(f"[Read more]({article['link']})")
                    else:
                        st.info("No recent news available.")
                except Exception:
                    st.info("Unable to fetch news at this time.")

    def render_technical_tab(self, config, df):
        """Render technical analysis tab"""
        st.markdown("## 🔍 Technical Analysis")

        # Support and resistance levels
        st.markdown("### 📍 Support & Resistance Levels")
        col1, col2 = st.columns(2)

        try:
            support, resistance = AdvancedIndicators.detect_support_resistance(df)

            with col1:
                st.markdown("**Support Levels:**")
                if support:
                    for level in support:
                        st.write(f"• {Formatter.currency(level)}")
                else:
                    st.write("No support levels detected")

            with col2:
                st.markdown("**Resistance Levels:**")
                if resistance:
                    for level in resistance:
                        st.write(f"• {Formatter.currency(level)}")
                else:
                    st.write("No resistance levels detected")
        except Exception:
            st.warning("Unable to calculate support/resistance levels")

        # Indicator charts
        st.markdown("### 📊 Technical Indicators")

        # MACD
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            try:
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'))
                if 'MACD_Diff' in df.columns:
                    fig_macd.add_trace(go.Bar(
                        x=df.index,
                        y=df['MACD_Diff'],
                        name='Histogram',
                        marker_color=np.where(df['MACD_Diff'] >= 0, 'green', 'red')
                    ))
                fig_macd.update_layout(title="MACD", height=300)
                st.plotly_chart(fig_macd, use_container_width=True)
            except Exception:
                st.warning("Unable to display MACD chart")

        # RSI
        if 'RSI_14' in df.columns:
            col1, col2 = st.columns([3, 1])
            with col1:
                try:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI'))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_yaxes(range=[0, 100])
                    fig_rsi.update_layout(title="RSI (14)", height=300)
                    st.plotly_chart(fig_rsi, use_container_width=True)
                except Exception:
                    st.warning("Unable to display RSI chart")

            with col2:
                st.markdown("**RSI Status:**")
                try:
                    last_rsi = df['RSI_14'].iloc[-1]
                    if pd.notna(last_rsi):
                        if last_rsi > 70:
                            st.error("Overbought (>70)")
                        elif last_rsi < 30:
                            st.success("Oversold (<30)")
                        else:
                            st.info("Neutral (30-70)")
                        st.metric("Current RSI", f"{last_rsi:.1f}")
                    else:
                        st.info("RSI data not available")
                except Exception:
                    st.info("RSI data not available")

    def render_comparison_tab(self, config, df, benchmark_df):
        """Render comparison tab"""
        st.markdown("## 🔄 Comparison Analysis")

        # Sector comparison
        st.markdown("### 📊 Sector Comparison")
        sector = st.selectbox(
            "Select sector for comparison",
            list(EnhancedDataService.SECTOR_ETFS.keys())
        )

        sector_etf = EnhancedDataService.SECTOR_ETFS[sector]
        sector_df = self.data_service.fetch_ohlcv(
            sector_etf,
            config['start'],
            config['end']
        )

        if not sector_df.empty and 'Close_Price' in sector_df.columns and 'Close_Price' in df.columns:
            # Normalized comparison
            try:
                comparison_df = pd.DataFrame({
                    config['symbol']: df['Close_Price'] / df['Close_Price'].iloc[0] * 100,
                    sector_etf: sector_df['Close_Price'] / sector_df['Close_Price'].iloc[0] * 100,
                    config['benchmark']: benchmark_df['Close_Price'] / benchmark_df['Close_Price'].iloc[
                        0] * 100 if not benchmark_df.empty and 'Close_Price' in benchmark_df.columns else pd.Series()
                }).dropna()

                if not comparison_df.empty and comparison_df.shape[1] > 1:
                    fig = go.Figure()
                    for col in comparison_df.columns:
                        fig.add_trace(go.Scatter(
                            x=comparison_df.index,
                            y=comparison_df[col],
                            name=col,
                            mode='lines'
                        ))

                    fig.update_layout(
                        title="Normalized Price Comparison (Base 100)",
                        yaxis_title="Normalized Price",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient data for comparison")
            except Exception:
                st.warning("Unable to create comparison chart")
        else:
            st.info("Sector data not available for comparison")

        # Correlation analysis
        st.markdown("### 📈 Correlation Matrix")

        # Get watchlist data
        symbols_to_compare = [config['symbol'], config['benchmark']] + config['watchlist'][
            :10]  # Limit to 10 for performance
        data_dict = {}

        with st.spinner("Loading comparison data..."):
            for symbol in symbols_to_compare:
                if symbol != config['symbol']:  # Already loaded
                    temp_df = self.data_service.fetch_ohlcv(symbol, config['start'], config['end'])
                    if not temp_df.empty:
                        data_dict[symbol] = temp_df
                else:
                    data_dict[symbol] = df

        # Add benchmark
        if not benchmark_df.empty:
            data_dict[config['benchmark']] = benchmark_df

        # Period selector for correlation
        corr_period = st.selectbox("Correlation Period", ["1M", "3M", "1Y", "All"], index=2)

        # Create correlation matrix
        try:
            corr_fig = self.visualizer.create_correlation_matrix(data_dict, corr_period)
            if corr_fig and corr_fig.data:
                st.plotly_chart(corr_fig, use_container_width=True)
            else:
                st.info("Not enough data to create correlation matrix")
        except Exception:
            st.warning("Unable to create correlation matrix")

    def render_risk_tab(self, config, df, metrics):
        """Render risk analysis tab"""
        st.markdown("## 📉 Risk Analysis")

        # Key risk metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Maximum Drawdown", Formatter.percentage(metrics.get('max_drawdown', 0)))

        with col2:
            st.metric("Value at Risk (95%)", Formatter.percentage(metrics.get('var_95', 0)))

        with col3:
            st.metric("Win Rate", Formatter.percentage(metrics.get('win_rate', 0)))

        with col4:
            st.metric("Sharpe Ratio", Formatter.number(metrics.get('sharpe_ratio', 0), 2))

        # Monte Carlo simulation
        if config['advanced']['monte_carlo']:
            st.markdown("### 🎲 Monte Carlo Simulation")

            col1, col2, col3 = st.columns(3)
            with col1:
                simulation_days = st.slider("Simulation Days", 30, 1000, 252)
            with col2:
                num_simulations = st.slider("Number of Simulations", 100, 5000, 1000)
            with col3:
                confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95)

            if st.button("Run Simulation", key="run_monte_carlo"):
                with st.spinner("Running Monte Carlo simulation..."):
                    try:
                        # Calculate parameters from historical data
                        log_returns = df['Log_Returns'].dropna()
                        if len(log_returns) > 10:
                            mu = log_returns.mean() * 252
                            sigma = log_returns.std() * np.sqrt(252)
                            initial_price = df['Close_Price'].iloc[-1]

                            # Run simulation
                            paths = AdvancedRiskMetrics.monte_carlo_simulation(
                                initial_price, mu, sigma, simulation_days, num_simulations
                            )

                            if paths.size > 0:
                                # Display results
                                fig = self.visualizer.create_monte_carlo_chart(
                                    paths, initial_price, confidence_level
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Statistics
                                final_prices = paths[-1, :]
                                st.markdown("#### Simulation Statistics")
                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    st.metric("Mean Final Price", Formatter.currency(np.mean(final_prices)))
                                with col2:
                                    st.metric("Median Final Price", Formatter.currency(np.median(final_prices)))
                                with col3:
                                    st.metric(
                                        "Probability of Gain",
                                        Formatter.percentage((final_prices > initial_price).mean() * 100)
                                    )
                                with col4:
                                    st.metric(
                                        "Expected Return",
                                        Formatter.percentage((np.mean(final_prices) / initial_price - 1) * 100)
                                    )
                            else:
                                st.warning("Simulation failed to generate results")
                        else:
                            st.warning("Insufficient historical data for simulation")
                    except Exception as e:
                        logger.error(f"Monte Carlo simulation error: {e}")
                        st.error("Error running Monte Carlo simulation")

        # Risk metrics table
        st.markdown("### 📊 Complete Risk Metrics")
        try:
            risk_metrics_df = pd.DataFrame({
                'Metric': [
                    'Total Return', 'Annual Return', 'Annual Volatility',
                    'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
                    'Max Drawdown', 'Average Drawdown', 'Value at Risk (95%)',
                    'Conditional VaR (95%)', 'Win Rate', 'Profit Factor',
                    'Skewness', 'Kurtosis', 'Beta', 'Alpha', 'R-squared'
                ],
                'Value': [
                    metrics.get('total_return', 0),
                    metrics.get('annual_return', 0),
                    metrics.get('annual_volatility', 0),
                    metrics.get('sharpe_ratio', 0),
                    metrics.get('sortino_ratio', 0),
                    metrics.get('calmar_ratio', 0),
                    metrics.get('max_drawdown', 0),
                    metrics.get('avg_drawdown', 0),
                    metrics.get('var_95', 0),
                    metrics.get('cvar_95', 0),
                    metrics.get('win_rate', 0),
                    metrics.get('profit_factor', 0),
                    metrics.get('skewness', 0),
                    metrics.get('kurtosis', 0),
                    metrics.get('beta', 0),
                    metrics.get('alpha', 0),
                    metrics.get('r_squared', 0)
                ]
            })

            # Format values
            def format_value(row):
                value = row['Value']
                metric = row['Metric']
                if pd.isna(value):
                    return "N/A"
                if any(x in metric for x in ['Return', 'Volatility', 'Drawdown', 'VaR', 'Rate']):
                    return Formatter.percentage(value)
                else:
                    return Formatter.number(value, 3)

            risk_metrics_df['Formatted'] = risk_metrics_df.apply(format_value, axis=1)

            st.dataframe(risk_metrics_df[['Metric', 'Formatted']], use_container_width=True)
        except Exception:
            st.warning("Unable to display complete risk metrics")

    def render_portfolio_tab(self, config, df):
        """Render portfolio analysis tab"""
        st.markdown("## 💼 Portfolio Analysis")

        if config['advanced']['portfolio_analysis']:
            # Get portfolio symbols
            portfolio_symbols = st.multiselect(
                "Select assets for portfolio",
                config['watchlist'] + [config['symbol'], config['benchmark']],
                default=[config['symbol'], config['benchmark']]
            )

            if len(portfolio_symbols) >= 2:
                # Fetch data for all symbols
                portfolio_data = {}
                with st.spinner("Loading portfolio data..."):
                    for symbol in portfolio_symbols:
                        if symbol == config['symbol']:
                            portfolio_data[symbol] = df
                        else:
                            temp_df = self.data_service.fetch_ohlcv(symbol, config['start'], config['end'])
                            if not temp_df.empty:
                                portfolio_data[symbol] = temp_df

                if len(portfolio_data) >= 2:
                    # Prepare returns data
                    returns_list = []
                    for symbol, symbol_df in portfolio_data.items():
                        if 'Returns' in symbol_df.columns:
                            returns_list.append(symbol_df['Returns'].rename(symbol))

                    if returns_list:
                        returns_df = pd.concat(returns_list, axis=1).dropna()

                        if not returns_df.empty and returns_df.shape[1] >= 2:
                            # Portfolio optimizer
                            st.markdown("### 🎯 Portfolio Optimization")

                            # Current weights (equal weight by default)
                            current_weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Current Weights:**")
                                for symbol, weight in zip(returns_df.columns, current_weights):
                                    st.write(f"{symbol}: {weight:.1%}")

                            with col2:
                                # Calculate current portfolio metrics
                                current_metrics = PortfolioAnalyzer.calculate_portfolio_metrics(
                                    current_weights,
                                    returns_df
                                )
                                st.metric("Current Sharpe", Formatter.number(current_metrics.get('sharpe_ratio', 0), 2))
                                st.metric("Current Volatility",
                                          Formatter.percentage(current_metrics.get('annual_volatility', 0)))

                            # Efficient frontier
                            if st.button("Generate Efficient Frontier", key="run_efficient_frontier"):
                                with st.spinner("Calculating efficient frontier..."):
                                    frontier_df = PortfolioAnalyzer.efficient_frontier(
                                        returns_df,
                                        num_portfolios=1000
                                    )

                                    if not frontier_df.empty:
                                        # Plot efficient frontier
                                        fig = px.scatter(
                                            frontier_df,
                                            x='volatility',
                                            y='return',
                                            color='sharpe',
                                            color_continuous_scale='viridis',
                                            title='Efficient Frontier'
                                        )

                                        # Mark current portfolio
                                        fig.add_trace(go.Scatter(
                                            x=[current_metrics.get('annual_volatility', 0)],
                                            y=[current_metrics.get('annual_return', 0)],
                                            mode='markers',
                                            marker=dict(size=15, color='red'),
                                            name='Current Portfolio'
                                        ))

                                        # Mark max Sharpe portfolio
                                        max_sharpe_idx = frontier_df['sharpe'].idxmax()
                                        fig.add_trace(go.Scatter(
                                            x=[frontier_df.loc[max_sharpe_idx, 'volatility']],
                                            y=[frontier_df.loc[max_sharpe_idx, 'return']],
                                            mode='markers',
                                            marker=dict(size=15, color='green', symbol='star'),
                                            name='Max Sharpe'
                                        ))

                                        fig.update_layout(height=500)
                                        st.plotly_chart(fig, use_container_width=True)

                                        # Show optimal portfolio weights
                                        st.markdown("#### 🏆 Optimal Portfolio (Max Sharpe Ratio)")
                                        optimal_weights = frontier_df.loc[max_sharpe_idx, 'weights']

                                        weights_df = pd.DataFrame({
                                            'Asset': returns_df.columns,
                                            'Weight': [f"{w:.1%}" for w in optimal_weights]
                                        })
                                        st.dataframe(weights_df, use_container_width=True)

                                        # Optimal portfolio metrics
                                        optimal_metrics = PortfolioAnalyzer.calculate_portfolio_metrics(
                                            optimal_weights,
                                            returns_df
                                        )

                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Expected Return",
                                                      Formatter.percentage(optimal_metrics.get('annual_return', 0)))
                                        with col2:
                                            st.metric("Expected Volatility",
                                                      Formatter.percentage(optimal_metrics.get('annual_volatility', 0)))
                                        with col3:
                                            st.metric("Sharpe Ratio",
                                                      Formatter.number(optimal_metrics.get('sharpe_ratio', 0), 2))
                                    else:
                                        st.warning("Unable to generate efficient frontier")
                        else:
                            st.warning("Insufficient returns data for portfolio analysis")
                    else:
                        st.warning("No returns data available for selected symbols")
                else:
                    st.warning("Unable to fetch data for all selected symbols")
            else:
                st.warning("Please select at least 2 assets for portfolio analysis")
        else:
            st.info("Enable 'Portfolio Analysis' in Advanced Features to use this tab.")

    def render_data_tab(self, config, df, info, metrics):
        """Render data and export tab"""
        st.markdown("## 📋 Data & Export")

        # Data preview
        st.markdown("### 📊 Data Preview")

        # Show last N rows
        preview_rows = st.slider("Rows to show", 10, 100, 20, key="preview_rows")
        try:
            st.dataframe(df.tail(preview_rows), use_container_width=True)
        except Exception:
            st.warning("Unable to display data preview")

        # Export options
        st.markdown("### 💾 Export Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Export CSV
            try:
                csv = df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{config['symbol']}_data.csv",
                    mime="text/csv"
                )
            except Exception:
                st.warning("Unable to generate CSV")

        with col2:
            # Export Excel
            @st.cache_data
            def to_excel(dataframe):
                from io import BytesIO
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='Stock Data')
                return output.getvalue()

            try:
                excel_data = to_excel(df)
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"{config['symbol']}_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception:
                st.warning("Unable to generate Excel file")

        with col3:
            # Export JSON
            try:
                # Limit data for JSON export
                export_df = df.tail(100).reset_index()
                export_df['Date'] = export_df.index.astype(str)
                json_data = export_df.to_json(orient='records', date_format='iso')
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"{config['symbol']}_data.json",
                    mime="application/json"
                )
            except Exception:
                st.warning("Unable to generate JSON")


# ---------- Main Execution ----------
if __name__ == "__main__":
    try:
        dashboard = EnhancedStockDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.exception("Application error")
        st.info(
            "Please refresh the page and try again. If the problem persists, try a different stock symbol or date range.")
