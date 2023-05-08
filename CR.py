import streamlit as st
import UtilLib as ul
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import yfinance as yf
from sklearn.linear_model import LinearRegression

###########
# Constants
###########
FN='CR.json'
LOOKBACK_WINDOW=90
FROM_YEAR=2022

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

def getWeightsDf():
  DT_FORMAT = '%d%b%y'
  lastUpdateDict = ul.jLoad('lastUpdateDict', FN)
  dts = []
  for v in lastUpdateDict.values():
    dts.append(datetime.datetime.strptime(v, DT_FORMAT))
  f = lambda dt: datetime.datetime.strftime(dt, DT_FORMAT)
  lastUpdate = f(np.max(dts))
  st.write(f"Last Update: {lastUpdate}")
  dts = [f(dt) for dt in dts]

  l = list()
  d = ul.jLoad('IBSDict', FN)
  ep = 1e-9
  ibsDict = {'SPY': 0,
             'QQQ': d['QQQ'] + ep,
             'TLT': d['TLT'] + ep,
             'IEF': 0,
             'GLD': 0,
             'UUP': 0}
  d = ul.jLoad('TPPDict', FN)
  tppDict = {'SPY': d['SPY'] + ep,
             'QQQ': d['QQQ'] + ep,
             'TLT': 0,
             'IEF': d['IEF'] + ep,
             'GLD': d['GLD'] + ep,
             'UUP': d['UUP'] + ep}
  i = 0
  for und in ['SPY', 'QQQ', 'TLT', 'IEF', 'GLD', 'UUP']:
    l.append([dts[i], und, (ibsDict[und] + tppDict[und]) / 2, ibsDict[und], tppDict[und]])
    i += 1
  df = pd.DataFrame(l)
  df.columns = ['Last Update', 'ETF', 'Total Weight', 'IBS (1/2)', 'TPP (1/2)']
  df.set_index(['ETF'], inplace=True)
  return df,lastUpdate

def getYFinanceS(ticker):
  from_date = f"{FROM_YEAR}-01-01"
  to_date = datetime.datetime.today().strftime('%Y-%m-%d')
  return yf.download(ticker, start=from_date, end=to_date)['Adj Close'].rename(ticker)

def getBeta(ts1, ts2, lookbackWindow):
  pcDf=ul.merge(ts1, ts2).pct_change().tail(lookbackWindow)
  regressor = LinearRegression(fit_intercept=False)
  X=pcDf.iloc[:,0].to_numpy().reshape(-1,1)
  y=pcDf.iloc[:,1].to_numpy().reshape(-1,1)
  #####
  regressor.fit(X,y)
  coef1 = regressor.coef_[0][0]
  mae1=np.mean(abs(X - y * coef1))
  #####
  regressor.fit(y,X)
  coef2 = 1/regressor.coef_[0][0]
  mae2 = np.mean(abs(X - y * coef2))
  #####
  return coef1 if mae1<mae2 else coef2

######
# Main
######
st.set_page_config(page_title='Core Reporter')
st.title('Core Reporter')

if checkPassword():

  # Weights
  st.header('Weights')
  df,lastUpdate=getWeightsDf()
  cols=['Total Weight','IBS (1/2)','TPP (1/2)']
  df[cols] = df[cols].applymap(lambda n:'' if n==0 else f"{n:.1%}")
  st.dataframe(df.style.apply(lambda row: ['background-color:red'] * len(row) if row['Last Update']==lastUpdate else [''] * len(row), axis=1))

  # Beta
  st.header('Beta (Return regressions of futures vs. ETFs)')
  tltTs = getYFinanceS('TLT')
  iefTs = getYFinanceS('IEF')
  zbTs = getYFinanceS('ZB=F')
  znTs = getYFinanceS('ZN=F')
  tnTs = getYFinanceS('TN=F')
  zb_tlt_beta=getBeta(zbTs, tltTs, LOOKBACK_WINDOW)
  st.write(f"ZB_TLT beta: {zb_tlt_beta:.3f} (Notional of futures to hold per 1x notional of ETF)")
  zn_ief_beta=getBeta(znTs, iefTs, LOOKBACK_WINDOW)
  st.write(f"ZN_IEF beta: {zn_ief_beta:.3f} (Notional of futures to hold per 1x notional of ETF)")
  tn_ief_beta=getBeta(tnTs, iefTs, LOOKBACK_WINDOW)
  st.write(f"TN_IEF beta: {tn_ief_beta:.3f} (Notional of futures to hold per 1x notional of ETF)")

  # Realized Performance
  st.header('Realized Performance')
  lastUpdate2=ul.jLoad('lastUpdateDict2',FN)
  st.write(f"Last Update: {lastUpdate2['realizedPerformance']}")
  st.write(f"MTD: {ul.jLoad('mtd',FN):.1%}")
  st.write(f"YTD: {ul.jLoad('ytd',FN):.1%}")

  # Backtest - Static
  st.header('Backtest - Static')
  st.write(f"Last Update: {lastUpdate2['backtestStatic']}")
  image = Image.open('BacktestStatic.png')
  st.image(image)
  st.markdown('YTD figures under **Realized Performance** can be different to those under **Backtest - Static** because of model changes implemented since the beginning of the year.')

  # Backtest - Live
  st.header('Backtest - Live')
  st.markdown('[Link](https://colab.research.google.com/drive/1dLe29LuqDhMy_MaIgl-OApsuIBwx3kCE?usp=sharing#forceEdit=true&sandboxMode=true)')

