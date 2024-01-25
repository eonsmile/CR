import QuantLib as ql
import UtilLib as ul
import streamlit as st

########
# Params
########
yrStart=ql.BTS_START_YEAR

######
# Main
######
z='BTS'
st.set_page_config(page_title=z)
st.title(z)
if ul.stCheckPW('password_BTS'):
  ql.runBTS(yrStart, isSkipTitle=True)
