# This is a Trade Opportunity Scan Tool
import streamlit as st
import yahoo_fin.stock_info as si

df1 = si.get_day_gainers()
df2 = si.get_day_losers()

# st.subheader("Top 100 Gainers")
st.markdown("<h2 style='text-align: center; color: Green;'>Top 100 Gainers</h2>", unsafe_allow_html=True)
# st.header(company_name + """ Analysis""")

st.write(df1.style.format({'Price (Intraday)': '${:,.2f}',
                           'Change': '${:,.2f}',
                           '% Change': '{:,.1f}%',
                           'Volume': '{:,.0f}',
                           'Avg Vol (3 month)': '{:,.0f}',
                           'Market Cap': '${:,.0f}',
                           'PE Ratio (TTM)': '{:,.2f}'}))
st.text('')
# st.subheader("Bottom 100 Stocks")
st.markdown("<h2 style='text-align: center; color: Red;'>Bottom 100 Losers</h2>", unsafe_allow_html=True)
st.write(df2.style.format({'Price (Intraday)': '${:,.2f}',
                           'Change': '${:,.2f}',
                           '% Change': '{:,.1f}%',
                           'Volume': '{:,.0f}',
                           'Avg Vol (3 month)': '{:,.0f}',
                           'Market Cap': '${:,.0f}',
                           'PE Ratio (TTM)': '{:,.2f}'}))
st.text('')
st.text('')
st.text('')
st.caption("ⓒ Franklin Chidi (FC) - MIT License")
