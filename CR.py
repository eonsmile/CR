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

def bt2016():
  st.divider()
  ql.runCore(2016)

######
# Init
######
if 'button_clicked' not in st.session_state:
  st.session_state.button_clicked = None

######
# Main
######
z='Core Reporter'
st.set_page_config(page_title=z)
st.title(z)

if ul.stCheckPW('password_CR'):
  # Weights
  st.header('Weights')
  df,lastUpdate=ql.getCoreWeightsDf()
  st.markdown(f"Last Update: <font color='red'>{lastUpdate}</font>", unsafe_allow_html=True)
  cols=['Total Weight','TPP (1/2)','RSS (1/4)','IBS (1/4)']
  df[cols] = df[cols].map(lambda n: '' if n == 0 else f"{n:.1%}")
  st.dataframe(df.style.apply(lambda row: ['background-color:red'] * len(row) if row['Last Update']==lastUpdate else [''] * len(row), axis=1))

  # Choices
  st.header('Choices')
  col1, col2, col3 = st.columns(3)
  with col1:
    if st.button('Betas'):
      st.session_state.button_clicked = 'betas'
  with col2:
    if st.button('Backtest (2016)'):
      st.session_state.button_clicked = '2016'
  with col3:
    if st.button('Backtest (2008)'):
      st.session_state.button_clicked = '2008'

  # Process
  if st.session_state.button_clicked == '2008':
    bt2008()
  elif st.session_state.button_clicked=='2016':
    bt2016()
  elif st.session_state.button_clicked=='betas':
    st.header('Betas (Return regressions of futures vs. ETFs)')
    d = ql.getCoreBetas()
    def m(label, beta):
      st.markdown(f"{label}: <font color='red'>{beta:.3f}</font>  (Notional of futures to hold per 1x notional of ETF)", unsafe_allow_html=True)
    m('ZN_IEF beta', d['ZN_IEF'])
    m('TN_IEF beta', d['TN_IEF'])


