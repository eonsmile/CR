###############
# Quant Library
###############
import UtilLib as ul
import streamlit as st
import numpy as np
import pandas as pd
import requests
import math
import pendulum
import yfinance as yf
from sklearn.linear_model import LinearRegression

###########
# Constants
###########
CC_API_KEY = st.secrets['cc_api_key']
START_YEAR_DICT={
  'priceHistory':2013-1,
  'YFinance':2023,
  'IBS':2013,
  'TPP':2013,
  'Core':2013,
}

#############################################################################################

###########
# Functions
###########
##########
# Backtest
##########
def bt(script,dp,dw,yrStart):
  st.header('Backtest')
  dp2 = dp.copy()
  dw2 = dw.copy()
  dwAllOrNone(dw2)
  validRows = ~dw2.isnull().any(axis=1)
  dtOrigin = dw2[validRows].index[np.where(dw2[validRows].index.year < yrStart)[0][-1]]
  dp2 = dp2.iloc[dp2.index >= dtOrigin]
  dw2 = dw2.iloc[dw2.index >= dtOrigin]
  ecS = dp2.iloc[:, 0].rename('Equity Curve') * 0
  ec = ecS.iloc[0] = 1
  p = dp2.iloc[0]
  w = dw2.iloc[0]
  for i in range(1, len(dp2)):
    r = dp2.iloc[i] / p - 1
    ecS.iloc[i] = ec * (1 + sum(w * r))
    if not dw2.iloc[i].isnull().any():
      w = dw2.iloc[i]
      p = dp2.iloc[i]
      ec = ecS.iloc[i]
  printCalendar(ecS)
  #####
  def m(s):
    d=dict()
    nYears = (s.index[-1] - s.index[0]).days / 365
    d['cagr'] = math.pow(s.iloc[-1] / s.iloc[0], 1 / nYears) - 1
    dd = s / s.cummax() - 1
    vol = ((np.log(s / s.shift(1)) ** 2).mean()) ** 0.5 * (252 ** 0.5)
    d['sharpe'] = d['cagr'] / vol
    d['mdd'] = -min(dd)
    d['mar'] = d['cagr']/d['mdd']
    return d
  #####
  d=m(ecS)
  d3=m(ecS[ecS.index>pendulum.instance(ecS.index[-1]).subtract(years=3).naive()])
  #####
  m=lambda label,z: f"{label}: <font color='red'>{z}</font>"
  sep='&nbsp;'*10
  st.markdown(sep.join([
    m('&nbsp;'*3+'Calmar', f"{d3['mar']:.2f}"),
    m('MAR', f"{d['mar']:.2f}"),
    m('Sharpe', f"{d['sharpe']:.2f}"),
    m('Cagr', f"{d['cagr']:.1%}"),
    m('MDD', f"{d['mdd']:.1%}"),
  ]), unsafe_allow_html=True)
  ul.cachePersist('w',script,ecS)

def btSetup(tickers, hvN=32, yrStart=START_YEAR_DICT['priceHistory'], applyDatesS=None):
  class m:
    def __init__(self, und,yrStart):
      self.und = und
      self.yrStart=yrStart
    #####
    def run(self):
      self.df=getPriceHistory(self.und,yrStart=self.yrStart)
      self.cS = self.df['Close'].rename(self.und)
  #####
  objs=[]
  for und in tickers:
    objs.append(m(und,yrStart))
  ul.parallelRun(objs)
  #####
  dfDict = dict()
  dp = None
  for obj in objs:
    dfDict[obj.und] = obj.df
    cS=obj.cS.to_frame()
    dp = cS if dp is None else ul.merge(dp, cS, how='outer')
  dp=dp.ffill()
  dw=dp.copy()
  dw[:] = np.nan
  hv = getHV(dp, n=hvN)
  if applyDatesS is None:
    return dp,dw,dfDict,hv
  else:
    return applyDates(dp, applyDatesS),applyDates(dw, applyDatesS),dfDict,applyDates(hv, applyDatesS)

def dwAllOrNone(dw):
  selection = dw.isnull().sum(axis=1).isin(list(range(1,len(dw.columns))))
  dw[selection] = dw.ffill()[selection]

def dwTail(dw,n=5):
  stWriteDf(dw.mask(dw.abs() == 0.0, 0.0).dropna().tail(n).round(3))

def printCalendar(s):
  def rgroup(r, groups):
    def rprod(n):
      return (n + 1).prod() - 1
    return r.groupby(groups).apply(rprod)
  #####
  r = s.pct_change()[1:]
  df = pd.DataFrame(rgroup(r, r.index.strftime('%Y-%m-01')))
  df.columns = ['Returns']
  df.index = df.index.map(pendulum.parse)
  df['Year'] = df.index.strftime('%Y')
  df['Month'] = df.index.strftime('%b')
  df = pd.pivot_table(data=df, index='Year', columns='Month', values='Returns', fill_value=0)
  df = df[ul.spl('Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec')]
  df['Year'] = rgroup(r, r.index.year).values
  df = df.map(lambda n: f"{n * 100:.1f}")
  height = (len(df)+1) * 35 + 3
  df = df.style.map(lambda z: f"color: {'red' if float(z) < 0 else '#228B22'}")
  st.dataframe(df,height=height)

#############################################################################################

#######
# Dates
#######
def applyDates(a,b):
  return a.reindex(b.index,method='pad').ffill().copy()

def endpoints(df, offset=0):
  ep_dates = pd.Series(df.index, index=df.index).resample('ME').max()
  date_idx = np.where(df.index.isin(ep_dates))
  date_idx = np.insert(date_idx, 0, 0)
  date_idx = np.append(date_idx, df.shape[0] - 1)
  if offset != 0:
    date_idx = date_idx + offset
    date_idx[date_idx < 0] = 0
    date_idx[date_idx > df.shape[0] - 1] = df.shape[0] - 1
  out = np.unique(date_idx)
  return out

#############################################################################################

#####
# Etc
#####
def cleanS(s, isMonthlyRebal=True):
  s=s.astype('float64').ffill()
  tmp=s.shift(1)
  if isinstance(s, pd.DataFrame):
    for i in range(1, len(s)):
      if s.iloc[i].equals(tmp.iloc[i]):
        s.iloc[i]=np.nan
  else:
    for i in range(1, len(s)):
      if s.iloc[i]==tmp.iloc[i]:
        s.iloc[i]=np.nan
  if isMonthlyRebal:
    pe=endpoints(s)
    s.iloc[pe]=s.ffill().iloc[pe]
  return s

def EMA(s, n):
  return s.ewm(span=n, min_periods=n, adjust=False).mean().rename('EMA')

def getBeta(ts1, ts2, lookbackWindow=90):
  pcDf=ul.merge(ts1, ts2,how='inner').pct_change().tail(lookbackWindow)
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

def getCoreBetas():
  tltS = getYFinanceS('TLT')
  iefS = getYFinanceS('IEF')
  zbS = getYFinanceS('ZB=F')
  znS = getYFinanceS('ZN=F')
  tnS = getYFinanceS('TN=F')
  zb_tlt_beta=getBeta(zbS, tltS)
  zn_ief_beta=getBeta(znS, iefS)
  tn_ief_beta=getBeta(tnS, iefS)
  return zb_tlt_beta,zn_ief_beta,tn_ief_beta

def getCoreWeightsDf():
  lastUpdateDict = ul.cachePersist('r','CR')['lastUpdateDict']
  fmt='DDMMMYY'
  dts = [pendulum.from_format(dt, fmt) for dt in lastUpdateDict.values()]
  lastUpdate = max(dts).format(fmt)

  l = list()
  d = ul.cachePersist('r', 'CR')['IBSDict']
  ep = 1e-9
  ibsDict = {'SPY': d['SPY'] + ep,
             'QQQ': d['QQQ'] + ep,
             'TLT': d['TLT'] + ep,
             'IEF': 0,
             'GLD': 0,
             'UUP': 0}
  d = ul.cachePersist('r', 'CR')['TPPDict']
  tppDict = {'SPY': d['SPY'] + ep,
             'QQQ': d['QQQ'] + ep,
             'TLT': 0,
             'IEF': d['IEF'] + ep,
             'GLD': d['GLD'] + ep,
             'UUP': d['UUP'] + ep}
  dts=list(lastUpdateDict.values())
  i = 0
  for und in ul.spl('SPY,QQQ,TLT,IEF,GLD,UUP'):
    l.append([dts[i], und, (ibsDict[und] + tppDict[und]) / 2, ibsDict[und], tppDict[und]])
    i += 1
  df = pd.DataFrame(l)
  df.columns = ul.spl('Last Update,ETF,Total Weight,IBS (1/2),TPP (1/2)')
  df.set_index(['ETF'], inplace=True)
  return df,lastUpdate

def getHV(s, n=32, af=252):
  if isinstance(s, pd.DataFrame):
    hv = s.copy()
    for col in hv.columns:
      hv[col].values[:] = getHV(hv[col], n=n, af=af)
    return hv
  else:
    variances= (np.log(s / s.shift(1))) ** 2
    return (EMA(variances,n)**.5*(af**.5)).rename(s.name)

def getIbsS(df):
  ibsS = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
  ibsS.rename('IBS',inplace=True)
  return ibsS

def getPriceHistory(und,yrStart=START_YEAR_DICT['priceHistory']):
  dtStart=str(yrStart)+ '-1-1'
  if und in ul.spl('BTC,ETH'):
    def m(toTs=None):
      z = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={und}&tsym=USD&limit=2000&api_key={CC_API_KEY}"
      if toTs is not None:
        z = f"{z}&toTs={toTs}"
      data = requests.get(z).json()['Data']
      return pd.DataFrame(data['Data']), data['TimeFrom']
    #####
    df, toTs = m()
    for i in range(2 if yrStart<2015 else 1):
      df2, toTs = m(toTs)
      df = pd.concat([df2.drop(df2.index[-1]), df])
    df['date'] = [pendulum.from_timestamp(s).naive() for s in df['time']]
    df = df[df['date'] > '2010-7-16']
    df['open'] = df['close'].shift()
    df = df[['date', 'open', 'high', 'low', 'close', 'volumefrom']]
  else: # EODHD
    ticker='VIX.INDX' if und=='VIX' else f"{und}.US"
    df=pd.DataFrame(requests.get(f"https://eodhd.com/api/eod/{ticker}?api_token={st.secrets['eodhd_api_key']}&fmt=json&from={dtStart}").json())
    df['date'] = pd.to_datetime(df['date'])
    df['ratio'] = df['adjusted_close'] / df['close']
    for field in ul.spl('open,high,low'):
      df[f"adjusted_{field}"] = df[field] * df['ratio']
    df = df[ul.spl('date,adjusted_open,adjusted_high,adjusted_low,adjusted_close,volume')]
  #####
  df = df.set_index('date')
  df.columns = ul.spl('Open,High,Low,Close,Volume')
  df = df.sort_values(by=['date']).round(10)
  return df

def getStateS(isEntryS, isExitS, isCleaned=False, isMonthlyRebal=True):
  if len(isEntryS)!=len(isExitS):
    ul.iExit('getStateS')
  stateS=(isEntryS * np.nan).rename('State')
  state=0
  for i in range(len(stateS)):
    if state==0 and isEntryS.iloc[i]:
      state=1
    if state==1 and isExitS.iloc[i]:
      state=0
    stateS.iloc[i]=state
  if isCleaned:
    stateS=cleanS(stateS, isMonthlyRebal=isMonthlyRebal)
  return stateS.astype(float)

def getYFinanceS(ticker, fromYear=START_YEAR_DICT['YFinance']):
  fromDate = f"{fromYear}-01-01"
  toDate = pendulum.today().format('YYYY-MM-DD')
  return yf.download(ticker, start=fromDate, end=toDate)['Adj Close'].rename(ticker)

def stWriteDf(df,isMaxHeight=False):
  def formatter(n):
    if isinstance(n,float):
      return f"{n:g}" if ~np.isnan(n) else ''
    else:
      return n
  #####
  df2 = df.copy()
  height=((len(df2) + 1) * 35 + 3)
  if isinstance(df2.index, pd.DatetimeIndex):
    df2.index = pd.to_datetime(df2.index).strftime('%Y-%m-%d')
  df2 = df2.replace(-0.0, 0.0).style.format(formatter)
  if 'State' in df2.columns:
    df2 = df2.map(lambda n: f"color: {'red' if n==0 else '#228B22'}", subset=['State'])
  if isMaxHeight:
    st.dataframe(df2, height=height)
  else:
    st.write(df2)

#############################################################################################

#########
# Scripts
#########
def runIBSCore(yrStart, multE=1, multQ=1, multB=1):
  def m(und, dfDict, isMondayS=None, sma200S=None):
    df = dfDict[und]
    ibsS = getIbsS(df)
    if und == undE:
      isEntryS = (isMondayS == 1) & (ibsS < .2) & (df['Low'] < df['Low'].shift(1))
      isExitS = df['Close'] > df['High'].shift(1)
    elif und == undQ:
      isEntryS = ibsS < .1
      isExitS = df['Close'] > df['High'].shift(1)
    elif und == undB:
      isEntryS = (df['Close']<sma200S) & (ibsS < .15)
      #isEntryS = (ibsS < .15) & (df['Low'] < df['Low'].shift(1))
      isExitS = ibsS > .55
    else:
      ul.iExit('runIBS')
    stateS = getStateS(isEntryS, isExitS, isCleaned=True, isMonthlyRebal=True)
    return ibsS, stateS
  #####
  undE = 'SPY'
  undQ = 'QQQ'
  undB = 'TLT'
  volTgt = .16
  maxWgt = 1
  dp, dw, dfDict, hv = btSetup([undE, undQ, undB],yrStart=yrStart-1)
  #####
  isMondayS = dfDict[undE]['Close'].rename('Monday?') * 0
  isMondayS[isMondayS.index.weekday == 0] = 1
  sma200S = dfDict[undB]['Close'].rolling(200).mean().rename('SMA200')
  ibsSE, stateSE = m(undE, dfDict, isMondayS=isMondayS)
  ibsSQ, stateSQ = m(undQ, dfDict)
  ibsSB, stateSB = m(undB, dfDict, sma200S=sma200S)
  dw[undE] = stateSE*multE
  dw[undQ] = stateSQ*multQ
  dw[undB] = stateSB*multB
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  dwAllOrNone(dw)
  d=dict()
  d['undE']=undE
  d['undQ']=undQ
  d['undB']=undB
  d['dp']=dp
  d['dw']=dw
  d['dfDict']=dfDict
  d['isMondayS']=isMondayS
  d['sma200S']=sma200S
  d['ibsSE']=ibsSE
  d['ibsSQ']=ibsSQ
  d['ibsSB']=ibsSB
  d['stateSE']=stateSE
  d['stateSQ']=stateSQ
  d['stateSB']=stateSB
  return d

def runIBS(yrStart,multE=1, multQ=1, multB=1,isSkipTitle=False):
  def m(d, und, ibsS, stateS, isMondayS=None, sma200S=None):
    df=d['dfDict'][und]
    st.subheader(und)
    df2 = ul.merge(df['Close'].round(2), df['High'].round(2), df['Low'].round(2), ibsS.round(3), how='inner')
    if isMondayS is not None: df2 = ul.merge(df2, isMondayS, how='inner')
    if sma200S is not None: df2 = ul.merge(df2, sma200S.round(2), how='inner')
    df2 = ul.merge(df2, stateS.ffill(), how='inner')
    stWriteDf(df2.tail())
  #####
  script = 'IBS'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runIBSCore(yrStart,multE=multE,multQ=multQ,multB=multB)
  st.header('Tables')
  m(d, d['undE'], d['ibsSE'], d['stateSE'], isMondayS=d['isMondayS'])
  m(d, d['undQ'], d['ibsSQ'], d['stateSQ'])
  m(d, d['undB'], d['ibsSB'], d['stateSB'], sma200S=d['sma200S'])
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runTPP(yrStart,multE=1,multQ=1,multB=1,multG=1,multD=1,isSkipTitle=False):
  undE = 'SPY'
  undQ = 'QQQ'
  undB = 'IEF'
  undG = 'GLD'
  undD = 'UUP'
  lookback = 32
  volTgt = .16
  maxWgt = 3
  ######
  script = 'TPP'
  if not isSkipTitle:
    st.header(script)
  ######
  dp, dw, dfDict, hv = btSetup([undE,undQ,undB,undG,undD],yrStart=yrStart-1)
  ratioDf = dp / dp.rolling(200).mean()
  isOkDf = (ratioDf >= 1) * 1
  wDf = (1 / hv) * isOkDf
  rDf = np.log(dp / dp.shift(1))
  for i in endpoints(rDf):
    origin = i - lookback + 1
    if origin >= 0:
      prS = rDf.iloc[origin:(i + 1)].multiply(wDf.iloc[i], axis=1).sum(axis=1)
      pHv = ((prS ** 2).mean()) ** .5 * (252 ** .5)
      dw.iloc[i] = wDf.iloc[i] * volTgt / pHv
  dw[undE]=dw[undE]*multE
  dw[undQ]=dw[undQ]*multQ
  dw[undB]=dw[undB]*multB
  dw[undG]=dw[undG]*multG
  dw[undD]=dw[undD]*multD
  dw.clip(0, maxWgt, inplace=True)
  st.header('Prices')
  stWriteDf(dp.tail())
  st.header('Ratios')
  stWriteDf(ratioDf.round(3).tail())
  st.header('Weights')
  dwTail(dw)
  bt(script, dp, dw, yrStart)

#####

def runAggregate(yrStart,strategies,weights,script):
  st.header(script)
  #####
  # Weights
  st.header('Weights')
  z = zip(strategies, weights)
  df = pd.DataFrame(z, columns=ul.spl('Strategy,Weight')).set_index('Strategy')
  stWriteDf(df)
  #####
  # Calcs
  dp = pd.DataFrame()
  for strategy in strategies:
    dp[strategy] = ul.cachePersist('r', strategy)
  dp = applyDates(dp, dp.iloc[:,-1]).ffill()
  dw = dp * np.nan
  pe = endpoints(dw)
  for i in range(len(weights)):
    dw.iloc[pe, i] = weights[i]
  #####
  # Backtest
  bt(script, dp, dw, yrStart)
  #####
  # Recent performance
  st.header('Recent Performance')
  dp2 = dp.copy()
  dp2[script] = ul.cachePersist('r', script)
  dp2 = dp2[[script] + strategies]
  dp2 = (dp2 / dp2.iloc[-1]).tail(23) * 100
  dp2 = dp2.round(2)
  stWriteDf(dp2, isMaxHeight=True)

def runCore(yrStart):
  runIBS(yrStart)
  st.divider()
  runTPP(yrStart)
  st.divider()
  strategies = ul.spl('IBS,TPP')
  weights = [1 / 2, 1 / 2]
  script = 'Core'
  runAggregate(yrStart, strategies, weights, script)