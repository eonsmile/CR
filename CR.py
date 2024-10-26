import streamlit as st
import QuantLib as ql
import UtilLib as ul

###########
# Functions
###########
def checkPassword():
  def m():
    isPWOk=st.session_state['pw'] == st.secrets['password_CR']
    st.session_state['isPWOk']=isPWOk
    if isPWOk: del st.session_state['pw']
  #####
  def m2():
    st.text_input('Password', type='password', on_change=m, key='pw')
  #####
  if 'isPWOk' not in st.session_state:
    m2()
    return False
  elif not st.session_state['isPWOk']:
    m2()
    st.error('😕 Password incorrect')
    return False
  else:    
    return True

def bt2008():
  st.divider()
  ql.runCore(2008)

def bt2015():
  st.divider()
  ql.runCore(2015)

######
# Init
######
if 'button_clicked' not in st.session_state:
  st.session_state.button_clicked = None

######
# Main
######
isOk=False
z='Core Reporter'
st.set_page_config(page_title=z)
st.title(z)

if ul.stCheckPW('password_CR'):
  # Weights
  st.header('Weights')
  df,lastUpdate=ql.getCoreWeightsDf()
  st.markdown(f"Last Update: <font color='red'>{lastUpdate}</font>", unsafe_allow_html=True)
  cols=['Total Weight','IBS (1/2)','TPP (1/2)']
  df[cols] = df[cols].map(lambda n: '' if n == 0 else f"{n:.1%}")
  st.dataframe(df.style.apply(lambda row: ['background-color:red'] * len(row) if row['Last Update']==lastUpdate else [''] * len(row), axis=1))

  # Beta
  st.header('Betas (Return regressions of futures vs. ETFs)')
  zb_tlt_beta, zn_ief_beta, tn_ief_beta = ql.getCoreBetas()
  def m(label, beta): st.markdown(f"{label}: <font color='red'>{beta:.3f}</font>  (Notional of futures to hold per 1x notional of ETF)", unsafe_allow_html=True)
  m('ZB_TLT beta',zb_tlt_beta)
  m('ZN_IEF beta',zn_ief_beta)
  m('TN_IEF beta',tn_ief_beta)
  isOk=True

  # Backtest
  st.header('Backtest')
  col1, col2 = st.columns(2)
  with col1:
    if st.button('From 2015'):
      st.session_state.button_clicked = '2015'
  with col2:
    if st.button('From 2008'):
      st.session_state.button_clicked = '2008'

  if st.session_state.button_clicked == '2008':
    bt2008()
  elif st.session_state.button_clicked=='2015':
    bt2013()

