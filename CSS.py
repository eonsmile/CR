import QuantLib as ql
import UtilLib as ul
import streamlit as st

########
# Params
########
yrStart=ql.CSS_START_YEAR

######
# Main
######
z='CSS'
st.set_page_config(page_title=z)
st.title(z)
if ul.stCheckPW('password_CSS'):
  ql.runCSS(yrStart, isSkipTitle=True)
