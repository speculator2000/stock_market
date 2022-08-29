#This is a Trade Opportunity Scan Tool
import streamlit as st
import yahoo_fin.stock_info as si

df1 = si.get_day_gainers()
df2 = si.get_day_losers()

st.write(df1)
st.write(df2)


