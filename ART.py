import QuantLib as ql
import UtilLib as ul
import streamlit as st

######
# Main
######
z='ART'
st.set_page_config(page_title=z)
st.title(z)

if ul.stCheckPW('password_CR'):
  l = ul.spl('SPY,QQQ,TLT,GLD')
  cols = st.columns(len(l))
  checkboxes = [cols[i].checkbox(l[i], value=True) for i in range(len(l))]
  multE=checkboxes[0]*1
  multQ=checkboxes[1]*1
  multB=checkboxes[2]*1
  multG=checkboxes[3]*1
  ql.runART(multE=multE,multQ=multQ,multB=multB,multG=multG,isSkipTitle=True)
