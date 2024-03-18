import QuantLib as ql
import UtilLib as ul
import streamlit as st

######
# Main
######
z='CSS'
st.set_page_config(page_title=z)
st.title(z)
#if ul.stCheckPW('password_CSS'):
ql.runCSS(isSkipTitle=True)
