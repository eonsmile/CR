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
  l = ul.spl('SPY,QQQ,TLT,GLD,FXI')
  chosenYear = st.radio('Start Year', ['2008', f"{yrStart}"], index=1)
  st.write('')
  cols = st.columns(len(l))
  checkboxes = [cols[i].checkbox(l[i], value=True) for i in range(len(l))]
  multE=checkboxes[0]*1
  multQ=checkboxes[1]*1
  multB=checkboxes[2]*1
  multG=checkboxes[3]*1
  multC=checkboxes[4]*1
  ql.START_YEAR_DICT['ART'] = int(chosenYear)
  ql.START_YEAR_DICT['priceHistory'] = ql.START_YEAR_DICT['ART'] - 1
  ql.runART(ql.START_YEAR_DICT['ART'],multE=multE,multQ=multQ,multB=multB,multG=multG,multC=multC,isSkipTitle=True)
