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
import pykalman
import yfinance as yf
import pandas_market_calendars
from sklearn.linear_model import LinearRegression
import pandas_ta

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
  'CSS':2013,
  'BTS':2015,
  'ART':2013
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

def getBizDayPriorNYSE(y,m,d):
  dt0=pendulum.date(y,m,d)
  dt=pandas_market_calendars.get_calendar('NYSE').schedule(start_date=dt0.subtract(days=8), end_date=dt0.subtract(days=1)).iloc[-1]['market_close']
  return pendulum.datetime(dt.year, dt.month, dt.day).naive()

def getBizDayNYSE(y,m,d,isAdjForward=True):
  dt0 = pendulum.date(y, m, d)
  if isAdjForward:
    dt = pandas_market_calendars.get_calendar('NYSE').schedule(start_date=dt0, end_date=dt0.add(days=8)).iloc[0]['market_close']
  else:
    dt = pandas_market_calendars.get_calendar('NYSE').schedule(start_date=dt0.subtract(days=8), end_date=dt0).iloc[-1]['market_close']
  return pendulum.datetime(dt.year, dt.month, dt.day).naive()

def getNYSECloseHourHKT():
  return 4 if pendulum.now('US/Eastern').is_dst() else 5

def getTodayNYSE():
  today = pendulum.today().naive()
  return getBizDayNYSE(today.year, today.month, today.day)

def getTomS(s, offsetBegin, offsetEnd, isNYSE=False): # 0,0 means hold one day starting from monthend
  s=s.copy()
  dtLast=s.index[-1]
  dtLast2=pendulum.instance(dtLast)
  if isNYSE:
    dts=pandas_market_calendars.get_calendar('NYSE').schedule(start_date=dtLast2, end_date=dtLast2.add(days=30)).index
  else:
    dts = [dtLast2.add(days=i).date() for i in range(30)]
    dts = pd.DatetimeIndex(pd.to_datetime(dts))
  s = s.reindex(s.index.union(dts))
  s[:]=0
  for i in range(offsetBegin, offsetEnd+1):
    s.iloc[endpoints(s, offset=i)]=1
  return s[s.index <= dtLast].rename('TOM?')

def getYestNYSE():
  today = pendulum.today().naive()
  yest=pandas_market_calendars.get_calendar('NYSE').schedule(start_date=today.subtract(days=8), end_date=today.subtract(days=1)).iloc[-1]['market_close']
  return pendulum.datetime(yest.year, yest.month, yest.day).naive()

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

def extend(df, df2):
  dtAnchor=df['Close'].first_valid_index()
  if df2.index[-1] >= dtAnchor:
    ratio= df.loc[dtAnchor]['Close'] / df2.loc[dtAnchor]['Close']
    df2= df2[:dtAnchor][:-1]
    df2[ul.spl('Open,High,Low,Close')] *= ratio
    df2['Volume'] /= ratio
    return pd.concat([df2, df.loc[dtAnchor:]])
  else:
    return df

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

def getKFMeans(s):
  kf = pykalman.KalmanFilter(n_dim_obs=1, n_dim_state=1,
                             initial_state_mean=0,
                             initial_state_covariance=1,
                             transition_matrices=[1],
                             observation_matrices=[1],
                             observation_covariance=1,
                             transition_covariance=0.05)
  means, _ = kf.filter(s)
  return pd.Series(means.flatten(), index=s.index)

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
  def m(und, dfDict, isMondayS=None):
    df = dfDict[und]
    ibsS = getIbsS(df)
    if und == undE:
      isEntryS = (isMondayS == 1) & (ibsS < .2) & (df['Low'] < df['Low'].shift(1))
      isExitS = df['Close'] > df['High'].shift(1)
    elif und == undQ:
      isEntryS = ibsS < .1
      isExitS = df['Close'] > df['High'].shift(1)
    elif und == undB:
      isEntryS = (ibsS < .15) & (df['Low'] < df['Low'].shift(1))
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
  ibsSE, stateSE = m(undE, dfDict, isMondayS=isMondayS)
  ibsSQ, stateSQ = m(undQ, dfDict)
  ibsSB, stateSB = m(undB, dfDict)
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
  d['ibsSE']=ibsSE
  d['ibsSQ']=ibsSQ
  d['ibsSB']=ibsSB
  d['stateSE']=stateSE
  d['stateSQ']=stateSQ
  d['stateSB']=stateSB
  return d

def runIBS(yrStart=START_YEAR_DICT['IBS'],multE=1, multQ=1, multB=1,isSkipTitle=False):
  def m(d, und, ibsS, stateS, isMondayS=None):
    df=d['dfDict'][und]
    st.subheader(und)
    df2 = ul.merge(df['Close'].round(2), df['High'].round(2), df['Low'].round(2), ibsS.round(3), how='inner')
    if isMondayS is not None: df2 = ul.merge(df2, isMondayS, how='inner')
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
  m(d, d['undB'], d['ibsSB'], d['stateSB'])
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runTPP(yrStart=START_YEAR_DICT['TPP']):
  tickers = ul.spl('SPY,QQQ,IEF,GLD,UUP')
  lookback = 32
  volTgt = .16
  maxWgt = 3
  ######
  script = 'TPP'
  st.header(script)
  ######
  dp, dw, dfDict, hv = btSetup(tickers,yrStart=yrStart-1)
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
  dw.clip(0, maxWgt, inplace=True)
  st.header('Prices')
  stWriteDf(dp.tail())
  st.header('Ratios')
  stWriteDf(ratioDf.round(3).tail())
  st.header('Weights')
  dwTail(dw)
  bt(script, dp, dw, yrStart)

def runBTSCore(yrStart):
  und = 'BTC'
  volTgt = .24
  maxWgt = 1
  df = getPriceHistory(und, yrStart=yrStart-1)
  dp = df[['Close']]
  dp.columns = [und]
  ratio1S = dp[und] / dp[und].shift(28)
  ratio1S.rename('Ratio 1', inplace=True)
  ratio2S = dp[und] / dp[und].rolling(5).mean()
  ratio2S.rename('Ratio 2', inplace=True)
  #####
  momScoreS = (ratio1S >= 1) * 1 + (ratio2S >= 1) * 1
  stateS = (momScoreS == 2) * 1
  stateS.rename('State', inplace=True)
  #####
  dw = dp.copy()
  dw[und] = stateS
  dw = cleanS(dw, isMonthlyRebal=True)
  hv = getHV(dp, n=16, af=365)
  dw = (dw * volTgt ** 2 / hv ** 2).clip(0, maxWgt)
  d=dict()
  d['und']=und
  d['dp'] = dp
  d['dw'] = dw
  d['ratio1S']=ratio1S
  d['ratio2S']=ratio2S
  d['stateS']=stateS
  return d

def runBTS(yrStart=START_YEAR_DICT['BTS'], isSkipTitle=False):
  script = 'BTS'
  if not isSkipTitle:
    st.header(script)
  d=runBTSCore(yrStart)
  st.header('Table')
  tableS = ul.merge(d['dp'][d['und']], d['ratio1S'].round(3), d['ratio2S'].round(3), d['stateS'], how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runARTCore(yrStart, multE=1, multQ=1, multG=1):
  volTgt = .08
  maxWgt = 1
  undE = 'SPY'
  undQ = 'QQQ'
  undG = 'GLD'
  tickers = [undE, undQ, undG]
  dp, dw, dfDict, hv = btSetup(tickers, yrStart=yrStart-1)
  #####
  hSE = dfDict[undE]['High']
  lSE = dfDict[undE]['Low']
  cSE = dfDict[undE]['Close']
  rSE = (cSE / cSE.shift() - 1).rename('Return')
  #####
  cSQ = dfDict[undQ]['Close']
  hSQ = dfDict[undQ]['High']
  #####
  hSG = dfDict[undG]['High']
  lSG = dfDict[undG]['Low']
  cSG = dfDict[undG]['Close']
  #####
  # SPY
  ratioSE = (cSE / cSE.rolling(200).mean()).rename('Ratio')
  key = 'sgArmor'
  s = ul.cachePersist('r', key, expireMins=60 * 12)
  if s is None:
    df = pd.read_csv('https://www.sumgrowth.com/StormGuardData.csv')
    df.index = pd.to_datetime(df['Date'])
    s = df['SG-Armor']
    ul.cachePersist('w', key, s)
  sgArmorSE = applyDates(s, cSE).shift()
  #####
  isEntryS = ((rSE < 0) & (rSE.shift().rolling(5).min() > 0) & (sgArmorSE > 0)) * 1
  isExitS = (cSE > hSE.shift()) * 1
  preState1SE = getStateS(isEntryS, isExitS, isCleaned=False, isMonthlyRebal=False).rename('Pre-State 1')
  #####
  wprSE = pandas_ta.willr(hSE, lSE, cSE, length=2).rename('WPR')
  isEntryS = ((wprSE < (-90)) & (sgArmorSE > 0))*1
  isExitS = (wprSE > (-10))*1
  preState2SE = getStateS(isEntryS, isExitS, isCleaned=False, isMonthlyRebal=False).rename('Pre-State 2')
  #####
  rsiSE = pandas_ta.rsi(cSE,length=2).rename('RSI')
  vixSE = applyDates(getPriceHistory('VIX',yrStart=START_YEAR_DICT['priceHistory'])['Close'],cSE).rename('VIX')
  vixSMA40SE = vixSE.rolling(40).mean().rename('VIX SMA40')
  vixSMA65SE = vixSE.rolling(65).mean().rename('VIX SMA65')
  isEntryS = ((rsiSE < 25) & (vixSMA40SE<vixSMA65SE) )*1
  isExitS = (rsiSE > 75)*1
  preState3SE = getStateS(isEntryS, isExitS, isCleaned=False, isMonthlyRebal=False).rename('Pre-State 3')
  #####
  sma6SE = cSE.rolling(6).mean().rename('SMA6')
  isEntryS=(rSE.rolling(2).min()>0) & (rSE==rSE.rolling(2).max()) & (ratioSE<1)
  isExitS=cSE<sma6SE
  preState4SE = -2*getStateS(isEntryS, isExitS, isCleaned=False, isMonthlyRebal=False).rename('Pre-State 4')
  #####
  stateSE = (preState1SE + preState2SE + preState3SE+preState4SE).clip(-1, 1).rename('State')
  #####
  # QQQ
  ratioSQ = (cSQ / cSQ.rolling(200).mean()).rename('Ratio')
  #####
  s=EMA(cSQ,130)/EMA(cSE,130)
  trigSQ=(s-s.shift()+0.0002).rename('Trig')*10000
  preState1SQ=((trigSQ>0) & (sgArmorSE>0))*1
  preState1SQ.rename('Pre-State 1',inplace=True)
  #####
  isTuesWedSQ = cSQ.rename('Tues/Wed?').astype(int) * 0
  isTuesWedSQ[isTuesWedSQ.index.weekday.isin([1, 2])] = 1
  isTwoDownDaysSQ = ((cSQ / cSQ.shift()).rolling(2).max()<1)*1
  isTwoDownDaysSQ.rename('Two Down Days?',inplace=True)
  isEntryS=(isTuesWedSQ==1)&(isTwoDownDaysSQ==1)&(sgArmorSE>0)
  isExitS=cSQ>hSQ.shift()
  preState2SQ=getStateS(isEntryS,isExitS,isCleaned=False, isMonthlyRebal=False).rename('Pre-State 2')
  #####
  ibsSQ = getIbsS(dfDict[undQ])
  isEntryS = (ratioSQ < 1) & ((cSQ / cSQ.shift(2)) > 1.02) & (cSQ > hSQ.shift())
  isExitS = ibsSQ < .2
  preState3SQ = -getStateS(isEntryS, isExitS, isCleaned=False, isMonthlyRebal=False).rename('Pre-State 3')
  stateSQ = (preState1SQ + preState2SQ + preState3SQ).clip(-1, 1).rename('State')
  #####
  # GLD
  cSB = applyDates(getPriceHistory('TLT')['Close'],cSG)
  cond1S = (cSG > hSG.rolling(3).max().shift())*1
  cond2S = (cSB > cSB.shift())*1
  cond3S = (cSG * 0).astype(int)
  cond3S.loc[cond3S.index.weekday != 3] = 1
  cond1S.rename('Conditon 1?',inplace=True)
  cond2S.rename('Conditon 2?',inplace=True)
  cond3S.rename('Conditon 3?',inplace=True)
  isEntryS = (cond1S & cond2S & cond3S) * 1
  isExitS = (cSG > hSG.shift()) * 1
  isExitS.loc[isEntryS == 1] = 0
  preState1SG = getStateS(isEntryS, isExitS, isCleaned=False, isMonthlyRebal=False).rename('Pre-State 1')
  #####
  ibsSG = getIbsS(dfDict[undG])
  adxSG = pandas_ta.adx(hSG, lSG, cSG, length=5)['ADX_5'].rename('ADX5')
  cond4S = ((ibsSG < .15) & (adxSG > 30) & (cSG.index.day>=15)) * 1
  cond4S.rename('Condition 4',inplace=True)
  isEntryS = cond4S
  isExitS = (cSG > cSG.shift()) * 1
  preState2SG = getStateS(isEntryS, isExitS, isCleaned=False, isMonthlyRebal=False).rename('Pre-State 2')
  #####
  stateSG=(preState1SG+preState2SG).clip(-1,1)
  stateSG.rename('State',inplace=True)
  #####
  # Summary
  dw[undE] = cleanS(stateSE, isMonthlyRebal=True) * multE
  dw[undQ] = cleanS(stateSQ, isMonthlyRebal=True) * multQ
  dw[undG] = cleanS(stateSG, isMonthlyRebal=True) * multG
  dw.loc[dw.index.year < yrStart] = 0
  dw = (dw * volTgt / hv).clip(-maxWgt, maxWgt)
  dwAllOrNone(dw)
  #####
  d=dict()
  d['undE']=undE
  d['undQ']=undQ
  d['undG']=undG
  d['dp'] = dp
  d['dw'] = dw
  d['dfDict'] = dfDict
  #####
  d['cSE'] = cSE
  d['rSE'] = rSE
  d['ratioSE'] = ratioSE
  d['sgArmorSE'] = sgArmorSE
  #####
  d['preState1SE']=preState1SE
  d['wprSE']=wprSE
  d['preState2SE']=preState2SE
  d['rsiSE']=rsiSE
  d['vixSE']=vixSE
  d['vixSMA40SE']=vixSMA40SE
  d['vixSMA65SE']=vixSMA65SE
  d['preState3SE']=preState3SE
  d['sma6SE']=sma6SE
  d['preState4SE']=preState4SE
  d['stateSE'] = stateSE
  #####
  d['cSQ'] = cSQ
  d['hSQ'] = hSQ
  d['ratioSQ']=ratioSQ
  #####
  d['trigSQ'] = trigSQ
  d['preState1SQ'] = preState1SQ
  d['isTuesWedSQ'] = isTuesWedSQ
  d['isTwoDownDaysSQ'] = isTwoDownDaysSQ
  d['preState2SQ'] = preState2SQ
  d['ibsSQ']=ibsSQ
  d['preState3SQ'] = preState3SQ
  d['stateSQ'] = stateSQ
  #####
  d['cSG'] = cSG
  d['hSG'] = hSG
  d['cond1S']=cond1S
  d['cond2S']=cond2S
  d['cond3S']=cond3S
  d['preState1SG']=preState1SG
  d['ibsSG']=ibsSG
  d['adxSG']=adxSG
  d['cond4S']=cond4S
  d['preState2SG']=preState2SG
  d['stateSG']=stateSG
  return d

def runART(yrStart=START_YEAR_DICT['ART'], multE=1, multQ=1, multG=1, isSkipTitle=False):
  script = 'ART'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runARTCore(yrStart,multE=multE,multQ=multQ,multG=multG)
  st.header('Tables')
  #####
  st.subheader(d['undE'])
  z=lambda n: f"{n:.1%}"
  tableSE = ul.merge(d['cSE'].round(2), d['rSE'].apply(z), d['ratioSE'].round(3), d['sgArmorSE'], d['preState1SE'], d['wprSE'].round(2), d['preState2SE'], how='inner')
  stWriteDf(tableSE.tail())
  tableSE2 = ul.merge(d['rsiSE'].round(2), d['vixSE'].round(2), d['vixSMA40SE'].round(2), d['vixSMA65SE'].round(2), d['preState3SE'], d['sma6SE'].round(2), d['preState4SE'],d['stateSE'].ffill(), how='inner')
  stWriteDf(tableSE2.tail())
  #####
  st.subheader(d['undQ'])
  tableSQ = ul.merge(d['cSQ'].round(2), d['hSQ'].round(2), d['ratioSQ'].round(3), d['trigSQ'].round(3), d['preState1SQ'], d['isTuesWedSQ'], d['isTwoDownDaysSQ'], d['preState2SQ'], how='inner')
  stWriteDf(tableSQ.tail())
  tableSQ2 = ul.merge(d['ibsSQ'], d['preState3SQ'], d['stateSQ'].ffill(), how='inner')
  stWriteDf(tableSQ2.tail())
  #####
  st.subheader(d['undG'])
  tableSG = ul.merge(d['cSG'].round(2), d['hSG'].round(2), d['cond1S'], d['cond2S'], d['cond3S'], d['preState1SG'], how='inner')
  stWriteDf(tableSG.tail())
  tableSG2 = ul.merge(d['ibsSG'].round(3), d['adxSG'].round(1), d['cond4S'], d['preState2SG'], d['stateSG'].ffill(), how='inner')
  stWriteDf(tableSG2.tail())
  #####
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

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

def runCore(yrStart=START_YEAR_DICT['Core']):
  runIBS()
  st.divider()
  runTPP()
  st.divider()
  strategies = ul.spl('IBS,TPP')
  weights = [1 / 2, 1 / 2]
  script = 'Core'
  runAggregate(yrStart, strategies, weights, script)