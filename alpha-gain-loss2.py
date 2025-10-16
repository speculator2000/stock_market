# Winners_Losers 2 - Trade Opportunity Scan Tool
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Winners & Losers",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Alpha Vantage API configuration
API_KEY = "JF4BRB377U85OERC"


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_top_gainers_losers():
    """Fetch top gainers and losers from Alpha Vantage API."""
    url = f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Check for API errors
        if "Error Message" in data:
            st.error(f"API Error: {data['Error Message']}")
            return None
        if "Note" in data:
            st.warning(f"API Note: {data['Note']}")
        if "Information" in data:
            st.info(f"API Information: {data['Information']}")

        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching data: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def create_dataframe(data, category):
    """Create a DataFrame from the API data with enhanced formatting."""
    if category not in data or not data[category]:
        return pd.DataFrame()

    df = pd.DataFrame(data[category])

    # Debug: Show available columns
    st.sidebar.write(f"Available columns in {category}: {list(df.columns)}")

    # Select and rename relevant columns - handle different column names
    column_mapping = {
        "ticker": "Symbol",
        "price": "Price",
        "change_amount": "Change",
        "change_percentage": "Percent Change",
        "change_percentage": "Percent Change",  # Handle potential variations
        "change_percentage": "Percent Change",
        "volume": "Volume"
    }

    # Create a mapping based on available columns
    available_mapping = {}
    for source_col, target_col in column_mapping.items():
        if source_col in df.columns:
            available_mapping[source_col] = target_col
        # Also check for variations in column names
        elif source_col.replace('_', ' ') in df.columns:
            available_mapping[source_col.replace('_', ' ')] = target_col

    # If no mapping found, try to infer columns
    if not available_mapping:
        st.warning(f"No standard columns found for {category}. Using available columns.")
        return df

    # Select only available columns and rename them
    df = df[list(available_mapping.keys())]
    df.columns = [available_mapping[col] for col in df.columns]

    # Convert numeric columns with proper handling
    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(',', ''), errors='coerce')

    if "Change" in df.columns:
        df["Change"] = pd.to_numeric(df["Change"].astype(str).str.replace(',', ''), errors='coerce')

    if "Percent Change" in df.columns:
        # Handle percentage values - remove % sign and convert to float
        df["Percent Change"] = pd.to_numeric(
            df["Percent Change"].astype(str).str.replace('%', '').str.replace(',', ''),
            errors='coerce'
        )

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"].astype(str).str.replace(',', ''), errors='coerce')

    return df


def sort_by_percentage_change(df, ascending=False):
    """Sort DataFrame by Percent Change column."""
    if df.empty or "Percent Change" not in df.columns:
        return df

    return df.sort_values("Percent Change", ascending=ascending)


def format_dataframe(df, title):
    """Apply formatting to the dataframe for better display."""
    if df.empty:
        return df

    # Create a copy to avoid modifying original
    styled_df = df.copy()

    # Format numeric columns with proper percentage handling
    if 'Price' in styled_df.columns:
        styled_df['Price'] = styled_df['Price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")

    if 'Change' in styled_df.columns:
        styled_df['Change'] = styled_df['Change'].apply(
            lambda x: f"{x:+,.2f}" if pd.notna(x) else "N/A"
        )

    if 'Percent Change' in styled_df.columns:
        # Format as percentage with one decimal point
        styled_df['Percent Change'] = styled_df['Percent Change'].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
        )

    if 'Volume' in styled_df.columns:
        styled_df['Volume'] = styled_df['Volume'].apply(
            lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
        )

    return styled_df


def create_performance_chart(gainers_df, losers_df):
    """Create visualization of top performers."""
    if gainers_df.empty and losers_df.empty:
        return None

    # Combine top 10 from each category for visualization
    top_gainers_viz = gainers_df.head(10).copy()
    top_losers_viz = losers_df.head(10).copy()

    top_gainers_viz['Category'] = 'Gainer'
    top_losers_viz['Category'] = 'Loser'

    combined_df = pd.concat([top_gainers_viz, top_losers_viz])

    if combined_df.empty:
        return None

    # Create bar chart
    fig = px.bar(
        combined_df,
        x='Symbol',
        y='Percent Change',
        color='Category',
        title='Top 10 Gainers vs Losers - Percentage Change',
        color_discrete_map={'Gainer': '#00ff00', 'Loser': '#ff0000'},
        text='Percent Change'
    )

    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )

    fig.update_layout(
        xaxis_title='Stock Symbol',
        yaxis_title='Percentage Change (%)',
        showlegend=True,
        hovermode='closest'
    )

    return fig


def create_volume_chart(gainers_df, losers_df):
    """Create volume comparison chart."""
    if gainers_df.empty and losers_df.empty:
        return None

    top_gainers_viz = gainers_df.head(10).copy()
    top_losers_viz = losers_df.head(10).copy()

    top_gainers_viz['Category'] = 'Gainer'
    top_losers_viz['Category'] = 'Loser'

    combined_df = pd.concat([top_gainers_viz, top_losers_viz])

    if combined_df.empty:
        return None

    fig = px.bar(
        combined_df,
        x='Symbol',
        y='Volume',
        color='Category',
        title='Trading Volume - Top 10 Gainers vs Losers',
        color_discrete_map={'Gainer': '#00ff00', 'Loser': '#ff0000'},
        log_y=True  # Use log scale for better visualization
    )

    fig.update_layout(
        xaxis_title='Stock Symbol',
        yaxis_title='Volume (Log Scale)',
        showlegend=True
    )

    return fig


def display_stock_metrics(gainers_df, losers_df):
    """Display key metrics about the market movers."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not gainers_df.empty and 'Percent Change' in gainers_df.columns:
            avg_gain = gainers_df['Percent Change'].mean()
            st.metric("Average Gain %", f"{avg_gain:.1f}%")
        else:
            st.metric("Average Gain %", "N/A")

    with col2:
        if not losers_df.empty and 'Percent Change' in losers_df.columns:
            avg_loss = losers_df['Percent Change'].mean()
            st.metric("Average Loss %", f"{avg_loss:.1f}%")
        else:
            st.metric("Average Loss %", "N/A")

    with col3:
        if not gainers_df.empty and 'Percent Change' in gainers_df.columns:
            max_gain = gainers_df['Percent Change'].max()
            st.metric("Max Gain %", f"{max_gain:.1f}%")
        else:
            st.metric("Max Gain %", "N/A")

    with col4:
        if not losers_df.empty and 'Percent Change' in losers_df.columns:
            max_loss = losers_df['Percent Change'].min()
            st.metric("Max Loss %", f"{max_loss:.1f}%")
        else:
            st.metric("Max Loss %", "N/A")


def main():
    # Header
    st.title("📈 Trade Opportunity Scanner")
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
    }
    .positive-change {
        color: green;
        font-weight: bold;
    }
    .negative-change {
        color: red;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("Real-time market movers analysis using Alpha Vantage API")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        auto_refresh = st.checkbox("Auto-refresh every 30 seconds", value=False)
        show_visualizations = st.checkbox("Show Visualizations", value=True)
        items_to_display = st.slider("Number of items to display", 10, 50, 25)

        st.header("📊 Sort Options")
        sort_gainers = st.selectbox(
            "Sort Gainers by:",
            ["Percent Change (Highest First)", "Percent Change (Lowest First)", "Volume", "Price"],
            index=0
        )
        sort_losers = st.selectbox(
            "Sort Losers by:",
            ["Percent Change (Lowest First)", "Percent Change (Highest First)", "Volume", "Price"],
            index=0
        )

        st.header("🔍 Debug Info")
        show_debug = st.checkbox("Show Debug Information", value=False)

        st.header("ℹ️ About")
        st.markdown("""
        This tool scans for:
        - **Top Gainers**: Stocks with highest percentage gains
        - **Top Losers**: Stocks with highest percentage losses
        - **Real-time data** from Alpha Vantage
        - **Auto-refresh** capability
        """)

        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")

    # Initialize session state
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None

    # Main content area
    col1, col2 = st.columns([1, 1])

    # Fetch and display data
    fetch_data = st.button("🔄 Fetch Latest Data",
                           type="primary") or auto_refresh or st.session_state.last_update is None

    if fetch_data:
        with st.spinner("Fetching real-time market data..."):
            data = fetch_top_gainers_losers()

            if data:
                # Debug: Show raw data structure
                if show_debug:
                    with st.expander("Raw API Data Structure"):
                        st.json(data)

                # Create DataFrames
                top_gainers_df = create_dataframe(data, "top_gainers")
                top_losers_df = create_dataframe(data, "top_losers")

                # Debug: Show dataframe info
                if show_debug:
                    st.write("Gainers DataFrame Info:")
                    st.write(top_gainers_df.info())
                    st.write("Losers DataFrame Info:")
                    st.write(top_losers_df.info())

                # Apply sorting based on user selection
                if not top_gainers_df.empty and 'Percent Change' in top_gainers_df.columns:
                    if sort_gainers == "Percent Change (Highest First)":
                        top_gainers_df = sort_by_percentage_change(top_gainers_df, ascending=False)
                    elif sort_gainers == "Percent Change (Lowest First)":
                        top_gainers_df = sort_by_percentage_change(top_gainers_df, ascending=True)
                    elif sort_gainers == "Volume" and 'Volume' in top_gainers_df.columns:
                        top_gainers_df = top_gainers_df.sort_values("Volume", ascending=False)
                    elif sort_gainers == "Price" and 'Price' in top_gainers_df.columns:
                        top_gainers_df = top_gainers_df.sort_values("Price", ascending=False)

                if not top_losers_df.empty and 'Percent Change' in top_losers_df.columns:
                    if sort_losers == "Percent Change (Lowest First)":
                        top_losers_df = sort_by_percentage_change(top_losers_df, ascending=True)
                    elif sort_losers == "Percent Change (Highest First)":
                        top_losers_df = sort_by_percentage_change(top_losers_df, ascending=False)
                    elif sort_losers == "Volume" and 'Volume' in top_losers_df.columns:
                        top_losers_df = top_losers_df.sort_values("Volume", ascending=False)
                    elif sort_losers == "Price" and 'Price' in top_losers_df.columns:
                        top_losers_df = top_losers_df.sort_values("Price", ascending=False)

                # Update session state
                st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.gainers_df = top_gainers_df
                st.session_state.losers_df = top_losers_df

                # Display update time
                st.success(f"✅ Data updated at {st.session_state.last_update}")

                # Display metrics
                display_stock_metrics(top_gainers_df, top_losers_df)

                # Display data tables
                with col1:
                    st.subheader("🏆 Top Gainers")
                    if not top_gainers_df.empty:
                        display_df = format_dataframe(top_gainers_df.head(items_to_display), "Gainers")

                        # Add custom styling for percentage changes
                        def style_percent_change(val):
                            if isinstance(val, str) and '%' in val:
                                if '+' in val:
                                    return 'color: green; font-weight: bold;'
                                elif '-' in val:
                                    return 'color: red; font-weight: bold;'
                            return ''

                        # Apply styling only if Percent Change column exists
                        if 'Percent Change' in display_df.columns:
                            styled_df = display_df.style.map(style_percent_change, subset=['Percent Change'])
                        else:
                            styled_df = display_df.style

                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                                "Price": st.column_config.TextColumn("Price", width="small"),
                                "Change": st.column_config.TextColumn("Change", width="small"),
                                "Percent Change": st.column_config.TextColumn("% Change", width="small"),
                                "Volume": st.column_config.TextColumn("Volume", width="medium")
                            }
                        )

                        # Show sorting info
                        st.caption(f"📊 Sorted by: {sort_gainers}")
                    else:
                        st.warning("No gainers data available")

                with col2:
                    st.subheader("📉 Top Losers")
                    if not top_losers_df.empty:
                        display_df = format_dataframe(top_losers_df.head(items_to_display), "Losers")

                        # Add custom styling for percentage changes
                        def style_percent_change(val):
                            if isinstance(val, str) and '%' in val:
                                if '+' in val:
                                    return 'color: green; font-weight: bold;'
                                elif '-' in val:
                                    return 'color: red; font-weight: bold;'
                            return ''

                        # Apply styling only if Percent Change column exists
                        if 'Percent Change' in display_df.columns:
                            styled_df = display_df.style.map(style_percent_change, subset=['Percent Change'])
                        else:
                            styled_df = display_df.style

                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                                "Price": st.column_config.TextColumn("Price", width="small"),
                                "Change": st.column_config.TextColumn("Change", width="small"),
                                "Percent Change": st.column_config.TextColumn("% Change", width="small"),
                                "Volume": st.column_config.TextColumn("Volume", width="medium")
                            }
                        )

                        # Show sorting info
                        st.caption(f"📊 Sorted by: {sort_losers}")
                    else:
                        st.warning("No losers data available")

                # Visualizations
                if show_visualizations and (not top_gainers_df.empty or not top_losers_df.empty):
                    st.subheader("📊 Market Analysis")

                    viz_col1, viz_col2 = st.columns(2)

                    with viz_col1:
                        perf_chart = create_performance_chart(top_gainers_df, top_losers_df)
                        if perf_chart:
                            st.plotly_chart(perf_chart, use_container_width=True)

                    with viz_col2:
                        vol_chart = create_volume_chart(top_gainers_df, top_losers_df)
                        if vol_chart:
                            st.plotly_chart(vol_chart, use_container_width=True)

                # Auto-refresh
                if auto_refresh:
                    time.sleep(30)
                    st.rerun()
            else:
                st.error("Failed to fetch data from Alpha Vantage API")

    elif st.session_state.last_update:
        # Display cached data
        st.info(f"📋 Showing cached data from {st.session_state.last_update}")

        gainers_df = st.session_state.get('gainers_df', pd.DataFrame())
        losers_df = st.session_state.get('losers_df', pd.DataFrame())

        if not gainers_df.empty or not losers_df.empty:
            display_stock_metrics(gainers_df, losers_df)

            with col1:
                st.subheader("🏆 Top Gainers")
                if not gainers_df.empty:
                    display_df = format_dataframe(gainers_df.head(items_to_display), "Gainers")

                    # Add custom styling for percentage changes
                    def style_percent_change(val):
                        if isinstance(val, str) and '%' in val:
                            if '+' in val:
                                return 'color: green; font-weight: bold;'
                            elif '-' in val:
                                return 'color: red; font-weight: bold;'
                        return ''

                    # Apply styling only if Percent Change column exists
                    if 'Percent Change' in display_df.columns:
                        styled_df = display_df.style.map(style_percent_change, subset=['Percent Change'])
                    else:
                        styled_df = display_df.style

                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    st.caption(f"📊 Sorted by: {sort_gainers}")
                else:
                    st.warning("No gainers data available")

            with col2:
                st.subheader("📉 Top Losers")
                if not losers_df.empty:
                    display_df = format_dataframe(losers_df.head(items_to_display), "Losers")

                    # Add custom styling for percentage changes
                    def style_percent_change(val):
                        if isinstance(val, str) and '%' in val:
                            if '+' in val:
                                return 'color: green; font-weight: bold;'
                            elif '-' in val:
                                return 'color: red; font-weight: bold;'
                        return ''

                    # Apply styling only if Percent Change column exists
                    if 'Percent Change' in display_df.columns:
                        styled_df = display_df.style.map(style_percent_change, subset=['Percent Change'])
                    else:
                        styled_df = display_df.style

                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    st.caption(f"📊 Sorted by: {sort_losers}")
                else:
                    st.warning("No losers data available")

    else:
        # Initial state
        st.info("👆 Click 'Fetch Latest Data' to load market movers")

    # Footer
    st.markdown("---")
    st.caption(
        f"ⓒ Trade Opportunity Scanner | Alpha Vantage API | Last Update: {st.session_state.last_update or 'Never'}")


if __name__ == "__main__":
    main()