import QuantLib as ql
import streamlit as st

# https://substack.com/home/post/p-162384852

######
# Main
######
z='VRS'
st.set_page_config(page_title=z)
st.title(z)

y = 2023
ql.runVRS(y, isSkipTitle=True)
