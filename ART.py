import QuantLib as ql
import UtilLib as ul
import streamlit as st

######
# Main
######
z='ART'
st.set_page_config(page_title=z)
st.title(z)
yrStart=ql.START_YEAR_DICT['ART']
if ul.stCheckPW('password_CR'):
  l = ul.spl('SPY,QQQ,GLD')
  chosenYear = st.radio('Start Year', ['2008', f"{yrStart}"], index=1)
  st.write('')
  cols = st.columns(len(l))
  checkboxes = [cols[i].checkbox(l[i], value=True) for i in range(len(l))]
  multE=checkboxes[0]*1
  multQ=checkboxes[1]*1
  multG=checkboxes[2]*1
  y = int(chosenYear)
  ql.START_YEAR_DICT['priceHistory'] = y - 1
  ql.runART(y,multE=multE,multQ=multQ,multG=multG,isSkipTitle=True)
