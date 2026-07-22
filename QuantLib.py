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
import yahooquery
import curl_cffi
import warnings
import json
import urllib.request
import pandas_ta_classic as ta

###########
# Constants
###########
SHARED_DICT={
  'yrStart':2016-1,
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
  dtFirstValid=dw2[validRows].index[0]
  if dtFirstValid.year < yrStart:
    dtOrigin = dw2[validRows].index[np.where(dw2[validRows].index.year < yrStart)[0][-1]]
  else:
    dtOrigin = dtFirstValid
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
    d['maxdd'] = -min(dd)
    d['mar'] = d['cagr']/d['maxdd']
    return d
  #####
  d=m(ecS)
  d3=m(ecS[ecS.index>pendulum.instance(ecS.index[-1]).subtract(years=3).naive()])
  #####
  m=lambda label,z: f"{label}: <font color='red'>{z}</font>"
  sep='&nbsp;'*10
  st.markdown(sep.join([
    m('&nbsp;' * 3 + 'Calmar', f"{d3['mar']:.2f}"),
    m('MAR', f"{d['mar']:.2f}"),
    m('Sharpe', f"{d['sharpe']:.2f}"),
    m('Cagr', f"{d['cagr']:.1%}"),
    m('MaxDD', f"{d['maxdd']:.1%}"),
  ]), unsafe_allow_html=True)
  ul.cachePersist('w',script,ecS)

def btSetup(tickers, hvN=32, yrStart=SHARED_DICT['yrStart']):
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
  return dp,dw,dfDict,hv

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

########
# Prices
########
#df2 = getPriceHistory('ITA', yrStart=yrStart)
#df2[['Close']].to_csv('tmp.csv', index_label='Date', date_format='%#m/%#d/%Y')

def getPriceHistory(und, yrStart=SHARED_DICT['yrStart']):
  dtStart=str(yrStart)+ '-1-1'
  if und.endswith('.T'):
    df = getPriceHistoryYahoo(und, yrStart=yrStart)
    df = df.loc[df.index >= pd.Timestamp(dtStart)]
  else:
    ticker=und
    if '.' not in ticker:
      ticker=f"{und}.US"
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
  #####
  def m(df,fn,sub=''):
    df2 = pd.read_csv(f"data/{sub}{fn}", index_col=0, parse_dates=True, date_format='%m/%d/%Y')
    for col in ['Open', 'High', 'Low', 'Volume']:
      df2[col] = df2['Close'] * (0 if col == 'Volume' else 1)
    return extend(df, df2)

  ###########################################################################
  # DECOMMISSIONED underlyings: no longer used by alpha. CSVs live in
  # data/Archive/. Logic is identical to original except the CSV path.
  ###########################################################################
  if und in ul.spl('EUDF.XETRA,IPRE.XETRA,COM,INFL,IBIT,'
                   'DFNS.LSE,DRAM,ENCO.LSE,GCOW,HFGM,JEGA.LSE,ORR,PFIX,'
                   'RARE.LSE,ROBO,ROLL.LSE,TAIL,WCOA.LSE,'
                   'COPX,GRID,WTAI.LSE,NATO.LSE,REMX,513A.T,9888.HK,9988.HK'):
    if und == 'EUDF.XETRA':
      dtStart = '2025-3-31'
    elif und == 'IPRE.XETRA':
      dtStart = '2018-12-28'
    # COM
    elif und=='INFL':
      dtStart = '2021-1-29'
    elif und=='DFNS.LSE':
      dtStart='2023-4-28'
    elif und=='DRAM':
      dtStart='2026-4-30'
    elif und=='ENCO.LSE':
      dtStart='2021-8-31'
    elif und=='GCOW':
      dtStart='2016-2-29'
    elif und=='HFGM':
      dtStart='2025-4-30'
    elif und == 'JEGA.LSE':
      dtStart = '2023-12-29'
    elif und=='ORR':
      dtStart = '2025-1-31'
    elif und=='PFIX':
      dtStart='2021-5-28'
    elif und=='RARE.LSE':
      dtStart='2024-4-30'
    elif und=='ROBO':
      dtStart = '2013-10-31'
    elif und=='ROLL.LSE':
      dtStart = '2020-12-29'
    elif und=='TAIL':
      dtStart = '2017-4-28'
    elif und=='WCOA.LSE':
      dtStart = '2025-9-30'
    elif und=='IBIT':
      dtStart = '2024-1-11'
    # ----- AIS / GEO legs (decommissioned), original dtStarts preserved -----
    elif und=='COPX':
      dtStart='2010-4-30'
    elif und=='GRID':
      dtStart='2009-11-30'
    elif und=='WTAI.LSE':
      dtStart = '2018-12-31'
    elif und == 'NATO.LSE':
      dtStart = '2023-7-31'
    elif und=='REMX':
      dtStart='2010-10-29'
    elif und=='513A.T':
      dtStart = '2026-2-27'
    elif und=='9888.HK':
      dtStart = '2021-3-23'
    elif und=='9988.HK':
      dtStart = '2019-11-26'
    else:
      dtStart = None
    if dtStart is not None: df = df.loc[df.index >= dtStart]
    df = m(df, f"{und}.csv", sub='Archive/')
    return df
  elif und == '000660.KO':
    return extend(getPriceHistoryIBKR('000660'), df.loc[df.index <= '2026-2-23'])
  elif und == 'DFND.SW':
    return m(df, 'ITA.csv', sub='Archive/')
  elif und == 'BDRY':
    return m(df, 'BDI.csv', sub='Archive/')    # GEO-only signal, now archived
  ###########################################################################
  # END decommissioned
  ###########################################################################

  if und in ul.spl('GDXJ,DBMF,PFMN.TO'):
    if und=='GDXJ':
      dtStart='2009-11-30'
    elif und=='DBMF':
      dtStart = '2019-5-31'
    elif und == 'PFMN.TO':
      dtStart = '2019-7-31'
    else:
      dtStart = None
    if dtStart is not None: df = df.loc[df.index >= dtStart]
    df = m(df, f"{und}.csv")
  elif und=='VIX1D.INDX':
    dtStart = '2023-4-24'
    df = df.loc[df.index>=dtStart]
  return df

'''
def getPriceHistoryCrypto(und, yrStart=SHARED_DICT['yrStart']):
  def m(toTs=None):
    z = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={und}&tsym=USD&limit=2000&api_key={st.secrets['cc_api_key']}"
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
  df = df.set_index('date')
  df.columns = ul.spl('Open,High,Low,Close,Volume')
  df = df.sort_values(by=['date']).round(10)
  return df
'''

def getPriceHistoryCrypto(und, yrStart=SHARED_DICT['yrStart']):
  dtStart = f"{max(yrStart, 2010)}-1-1"
  url = f"https://eodhd.com/api/eod/{und}-USD.CC?api_token={st.secrets['eodhd_api_key']}&fmt=json&from={dtStart}"
  df = pd.DataFrame(requests.get(url).json())
  df['date'] = pd.to_datetime(df['date'])
  df = df[df['date'] > '2010-7-16']
  df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
  df = df.set_index('date')
  df.columns = ul.spl('Open,High,Low,Close,Volume')
  df = df.sort_values(by=['date']).round(10)
  return df

def getPriceHistoryFred(id, yrStart=SHARED_DICT['yrStart']):
  dtStart = f"{yrStart}-01-01"
  url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd={dtStart}"
  df = pd.read_csv(url, parse_dates=['observation_date'], index_col='observation_date')
  df.columns = [id]
  df = df.replace('.', np.nan).astype(float).dropna()
  df.index.name = None
  return df[id]

def getPriceHistoryIBKR(und):
  from ib_async import IB, Stock
  import logging
  import pandas as pd

  if isinstance(und, str) and und.isdigit() and len(und) == 6:
    contract = Stock(und, 'KRX', 'KRW')
  else:
    contract = Stock(und, 'SMART', 'USD')

  ib_logger = logging.getLogger('ib_async')
  prev_level = ib_logger.level
  ib_logger.setLevel(logging.CRITICAL)

  ib = IB()
  try:
    for port in (4001, 7496):
      try:
        ib.connect('127.0.0.1', port, clientId=1, readonly=True, timeout=5)
        if ib.isConnected():
          break
      except Exception:
        continue
    else:
      raise ConnectionError('Could not connect to IB Gateway (4001) or TWS (7496)')
  finally:
    ib_logger.setLevel(prev_level)

  ib.reqMarketDataType(3)

  bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='2 Y',
    barSizeSetting='1 day',
    whatToShow='ADJUSTED_LAST',
    useRTH=True,
    formatDate=1,
  )

  ib.disconnect()

  df = pd.DataFrame([{
    'Date':   b.date,
    'Open':   b.open,
    'High':   b.high,
    'Low':    b.low,
    'Close':  b.close,
    'Volume': b.volume,
  } for b in bars])

  if df.empty:
    return df

  df['Date'] = pd.to_datetime(df['Date'])
  df = df.set_index('Date').sort_index()
  return df.loc['2026-01-01':]

def getPriceHistoryYahoo(und, yrStart=SHARED_DICT['yrStart']):
  dtStart = str(yrStart) + '-1-1'
  period = max(2, pendulum.now().year - yrStart + 1)
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning)
    session = curl_cffi.Session(impersonate="chrome")
    df = yahooquery.Ticker(und, session=session).history(period=f"{period}y")
  df.index = df.index.droplevel('symbol')
  df.index = pd.DatetimeIndex(pd.to_datetime(
    [pendulum.parse(str(x)).date() for x in df.index]
  )).tz_localize(None).normalize()
  df.index.name = 'date'
  ratio = df['adjclose'] / df['close']
  df['open'] = df['open'] * ratio
  df['high'] = df['high'] * ratio
  df['low'] = df['low'] * ratio
  df['close'] = df['adjclose']
  df = df[['open', 'high', 'low', 'close', 'volume']]
  df.columns = ul.spl('Open,High,Low,Close,Volume')
  df = df.sort_index().round(10)
  df = df[~df.index.duplicated(keep='last')]
  df = df.loc[df.index >= pd.Timestamp(dtStart)]
  return df

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

def getCoreWeightsDf():
  lastUpdateDict = ul.cachePersist('r','CR')['lastUpdateDict']
  fmt='DDMMMYY'
  dts = [pendulum.from_format(dt, fmt) for dt in lastUpdateDict.values()]
  lastUpdate = max(dts).format(fmt)
  l = list()
  ep = 1e-9
  #####
  emptyDict = {'SPY':0,'QQQ':0,'IEI':0,'GLD':0,'UUP':0}
  #####
  d = ul.cachePersist('r', 'CR')['TPPDict']
  tppDict=emptyDict.copy()
  for und in ul.spl('QQQ,IEI,GLD,UUP'):
    tppDict[und] = d[und] + ep
  ####
  d = ul.cachePersist('r', 'CR')['RSSDict']
  rssDict = emptyDict.copy()
  rssDict['SPY'] = d['SPY'] + ep
  ####
  d = ul.cachePersist('r', 'CR')['IBSDict']
  ibsDict = emptyDict.copy()
  ibsDict['QQQ'] = d['QQQ'] + ep
  #####
  dts=list(lastUpdateDict.values())
  i = 0
  for und in ul.spl('SPY,QQQ,IEI,GLD,UUP'):
    totalWeight = tppDict[und]/3+rssDict[und]/3+ibsDict[und]/3
    l.append([dts[i], und, totalWeight, tppDict[und], rssDict[und], ibsDict[und]])
    i += 1
  df = pd.DataFrame(l)
  df.columns = ul.spl('Last Update,ETF,Total Weight,TPP (1/3),RSS (1/3),IBS (1/3)')
  df.set_index(['ETF'], inplace=True)
  return df,lastUpdate

def getHV(s, n=32, af=252):
  if isinstance(s, pd.DataFrame):
    hv = s.copy()
    for col in hv.columns:
      hv[col] = getHV(hv[col], n=n, af=af).values
    return hv
  else:
    variances= (np.log(s / s.shift(1))) ** 2
    return (EMA(variances,n)**.5*(af**.5)).rename(s.name)

def getIbsS(df,n=1):
  if n==1:
    ibsS = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
  else:
    lS=df['Low'].rolling(n).min()
    hS=df['High'].rolling(n).max()
    ibsS = (df['Close'] - lS) / (hS - lS)
  ibsS.name = 'IBS'
  return ibsS

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

def getStateS_timestop(isEntryS, isExitS, maxDays, isCleaned=False, isMonthlyRebal=True):
  if len(isEntryS)!=len(isExitS):
    ul.iExit('getStateS_timestop')
  stateS=(isEntryS * np.nan).rename('State')
  state=0
  daysHeld=0
  for i in range(len(stateS)):
    if state==0 and isEntryS.iloc[i]:
      state=1; daysHeld=0
    if state==1:
      if isExitS.iloc[i] or daysHeld>=maxDays: state=0
      else: daysHeld+=1
    stateS.iloc[i]=state
  if isCleaned:
    stateS=cleanS(stateS, isMonthlyRebal=isMonthlyRebal)
  return stateS.astype(float)

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
  with pd.option_context('future.no_silent_downcasting', True):
    df2 = df2.replace(-0.0, 0.0)
  df2 = df2.style.format(formatter)
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
def runIBSCore(yrStart):
  und = 'QQQ'
  volTgt = .225
  maxWgt = 1.5
  dp, dw, dfDict, hv = btSetup([und],yrStart=yrStart-1)
  #####
  df = dfDict[und]
  ibsS = getIbsS(df)
  isEntryS = ibsS < .1
  isExitS = (ibsS > .9) | (df['Close'] > df['High'].shift(1))
  stateS = getStateS_timestop(isEntryS, isExitS, 7, isCleaned=True, isMonthlyRebal=True)
  dw[und] = stateS
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  dwAllOrNone(dw)
  d=dict()
  d['dp']=dp
  d['dw']=dw
  d['dfDict']=dfDict
  d['ibsS']=ibsS
  d['stateS']=stateS
  return d

def runIBS(yrStart,isSkipTitle=False):
  script = 'IBS'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runIBSCore(yrStart)
  st.header('Table')
  df = d['dfDict']['QQQ']
  df2 = ul.merge(df['Close'].round(2), df['High'].round(2), df['Low'].round(2), d['ibsS'].round(3), how='inner')
  df2 = ul.merge(df2, d['stateS'].ffill(), how='inner')
  stWriteDf(df2.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runRSSCore(yrStart):
  und='SPY'
  dp, dw, dfDict, hv = btSetup([und],yrStart=yrStart-1)
  #####
  cS = dfDict[und]['Close']
  vixS = applyDates(getPriceHistory('VIX.INDX',yrStart=yrStart-1)['Close'].rename('VIX'),cS)
  rsiS = ta.rsi(cS, length=2).rename('RSI2')
  ibsS = getIbsS(dfDict[und],4)
  ratioS = (cS/cS.rolling(200).mean()).rename('Ratio')
  vixRatioS = (vixS.rolling(40).mean()/vixS.rolling(65).mean()).rename('VIX Ratio')
  #####
  isEntryS = (rsiS < 25) & (ibsS<.3) & (ratioS>1) & (vixRatioS<1)
  isExitS = (rsiS > 75) | (ibsS>.7) | (ratioS<=1) | (vixRatioS>=1)
  stateS = getStateS(isEntryS, isExitS, isCleaned=True, isMonthlyRebal=True)
  #####
  # Summary
  dw[und] = stateS * 2
  dw.loc[dw.index.year < yrStart] = 0
  d=dict()
  d['dp']=dp
  d['dw']=dw
  d['vixS']=vixS
  d['rsiS']=rsiS
  d['ibsS']=ibsS
  d['ratioS']=ratioS
  d['vixRatioS']=vixRatioS
  d['stateS']=stateS
  return d

def runRSS(yrStart,isSkipTitle=False):
  script = 'RSS'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runRSSCore(yrStart)
  st.header('Table')
  tableS = ul.merge(d['dp'],d['rsiS'].round(1),d['ibsS'].round(3),d['ratioS'].round(3),d['vixS'],d['vixRatioS'].round(3), d['stateS'].ffill(), how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runJMRCore(yrStart):
  etc=['DXJ']
  dp, dw, dfDict, hv = btSetup(ul.spl('EWJ,FXY')+etc,yrStart=yrStart-1)
  dp2 = dp.copy()
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
    hv = hv.drop(und2, axis=1)
  #####
  ibs_DXJ = getIbsS(dfDict['DXJ']).rename('IBS DXJ')
  ibs_EWJ   = getIbsS(dfDict['EWJ']).rename('IBS EWJ')
  isEntryS = (ibs_DXJ < .15) & (ibs_EWJ < .3)
  isExitS  = ibs_DXJ > .7
  stateS   = getStateS_timestop(isEntryS, isExitS, 7, isCleaned=True, isMonthlyRebal=True)
  #####
  dw['EWJ'] = stateS * 1.65
  dw['FXY'] = -dw['EWJ']
  #####
  d=dict()
  d['dp']=dp
  d['dp2']=dp2
  d['dw']=dw
  d['dfDict']=dfDict
  d['ibs_DXJ']=ibs_DXJ
  d['ibs_EWJ']=ibs_EWJ
  d['stateS']=stateS
  return d

def runJMR(yrStart,isSkipTitle=False):
  script = 'JMR'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runJMRCore(yrStart)
  st.header('Table')
  df2 = ul.merge(d['dp2'],d['ibs_DXJ'].round(3), d['ibs_EWJ'].round(3), how='inner')
  df2 = ul.merge(df2, d['stateS'].ffill(), how='inner')
  stWriteDf(df2.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runTPP(yrStart,multQ=1,multB=1,multG=1,multD=1,isSkipTitle=False):
  undQ = 'QQQ'
  undB = 'IEI'
  undG = 'GLD'
  undD = 'UUP'
  lookback = 32
  volTgt = .155
  maxWgt = 1.5
  ######
  script = 'TPP'
  if not isSkipTitle:
    st.header(script)
  ######
  dp, dw, dfDict, hv = btSetup([undQ,undB,undG,undD],yrStart=yrStart-1)
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

def runTPP2Core(yrStart):
  volTgt = .155
  maxWgt = 1.5
  etc=ul.spl('HYG,SPHB,SPLV,TIP')
  dp, dw, dfDict, hv = btSetup(ul.spl('SPY,GLD,UUP')+etc,yrStart=yrStart-1)
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
    hv = hv.drop(und2, axis=1)
  ##############
  # SPY canaries
  ##############
  # 1. BTC
  btcS = getPriceHistoryCrypto('BTC', yrStart - 1)['Close'].rename('BTC')
  ratio50S_BTC = (btcS / btcS.rolling(50).mean()).rename('BTC Ratio 50D')
  isCanaryS_BTC = applyDates(ratio50S_BTC > 1, dp).rename('BTC') * 1.0

  # 2. HYG
  hygS = applyDates(dfDict['HYG']['Close'],dp)
  ratio100S_HYG = (hygS / EMA(hygS, 100)).rename('HYG Ratio 100D')
  isCanaryS_HYG = (ratio100S_HYG > 1).rename('HYG') * 1.0

  # 3. SPHB/LV momentum
  def m(s):
    sum = 0
    for i in range(13):
      sum += s.shift(i * 21)
    return 13 * s / sum - 1
  #####
  sphbS = applyDates(dfDict['SPHB']['Close'],dp)
  splvS = applyDates(dfDict['SPLV']['Close'],dp)
  sphb_lv_ratio = sphbS / splvS
  isCanaryS_SPHB_LV = (m(sphb_lv_ratio) > 0).rename('SPHB_LV') * 1.0

  # 4. TIP
  cS_TIP = applyDates(dfDict['TIP']['Close'],dp)
  momS_TIP = (cS_TIP.pct_change(21) + cS_TIP.pct_change(63) + cS_TIP.pct_change(126) + cS_TIP.pct_change(252)) / 4
  isCanaryS_TIP = (momS_TIP > 0).rename('TIP') * 1.0

  # Voting
  voteDf = pd.DataFrame({
    'BTC': isCanaryS_BTC,
    'HYG': isCanaryS_HYG,
    'SPHB_LV': isCanaryS_SPHB_LV,
    'TIP': isCanaryS_TIP,
  })
  voteCountS = voteDf.sum(axis=1).rename('Votes')

  #########
  # UUP/GLD
  #########
  gldS = applyDates(dfDict['GLD']['Close'], dp)
  uupS = applyDates(dfDict['UUP']['Close'], dp)
  ratio150S_GLD = (gldS / gldS.rolling(150).mean()).rename('GLD Ratio 150D')
  ratio50S_UUP = (uupS / uupS.rolling(50).mean()).rename('UUP Ratio 50D')
  #####
  m = lambda n: applyDates(n, dw) * 1
  dw['SPY'] = (voteCountS >= 2) * (voteCountS / 4)
  dw['GLD'] = m(ratio150S_GLD>1)
  dw['UUP'] = m(ratio50S_UUP>1)
  #####
  stateDf=dw.astype(float).ffill()
  dw=cleanS(dw,isMonthlyRebal=True)
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  #####
  d=dict()
  d['dp']=dp
  d['dw']=dw
  d['stateDf']=stateDf
  d['btcS'] = btcS
  d['voteDf'] = voteDf
  d['voteCountS'] = voteCountS
  d['isCanaryS_BTC'] = isCanaryS_BTC
  d['isCanaryS_HYG'] = isCanaryS_HYG
  d['isCanaryS_SPHB_LV'] = isCanaryS_SPHB_LV
  d['isCanaryS_TIP'] = isCanaryS_TIP
  #####
  d['ratio150S_GLD']=ratio150S_GLD
  d['ratio50S_UUP']=ratio50S_UUP
  return d

def runTPP2(yrStart, isSkipTitle=False):
  script = 'TPP2'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runTPP2Core(yrStart)
  dp=d['dp']
  st.header('Prices')
  stWriteDf(ul.merge(dp,d['btcS'],how='inner').tail())
  st.header('SPY Canaries')
  stWriteDf(ul.merge(
    d['isCanaryS_BTC'], d['isCanaryS_HYG'],
    d['isCanaryS_SPHB_LV'], d['isCanaryS_TIP'],
    d['voteCountS'], how='inner').tail())
  st.header('GLD/UUP Ratios')
  stWriteDf(ul.merge(applyDates(d['ratio150S_GLD'], dp).round(3), applyDates(d['ratio50S_UUP'], dp).round(3), how='inner').tail())
  st.header('States')
  stWriteDf(d['stateDf'].tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runBTSCore(yrStart):
  volTgt = .255
  maxWgt = 1.5
  cS = getPriceHistoryCrypto('BTC', yrStart=yrStart)['Close']
  ratioS = (cS/cS.rolling(50).mean()).rename('Ratio')
  ratio2S = (cS / cS.rolling(240).max()).rename('Ratio2')
  dw = ((ratioS>1) & (ratio2S>0.8)).rename('BTC').to_frame()
  dp = cS.rename('BTC').to_frame()
  hv = getHV(dp, af=365)
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  dw = cleanS(dw, isMonthlyRebal=True)
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['ratioS'] = ratioS
  d['ratio2S'] = ratio2S
  return d

def runBTS(yrStart, isSkipTitle=False):
  script = 'BTS'
  if not isSkipTitle:
    st.header(script)
  d = runBTSCore(yrStart)
  st.header('Table')
  tableS = ul.merge(d['dp'], d['ratioS'].round(3), d['ratio2S'].round(3), how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runVTSCore(yrStart):
  volTgt = .28
  maxWgt = 1.5
  #####
  und = 'SPY'
  dp, dw, dfDict, hv = btSetup([und], yrStart=yrStart - 1)
  #####
  cS = dfDict[und]['Close']
  vixS = applyDates(getPriceHistory('VIX.INDX', yrStart=yrStart - 1)['Close'].rename('VIX'), cS)
  vixRatioS = (vixS / vixS.rolling(10).mean()).rename('VIX Ratio')
  #####
  isVixOk = vixRatioS < 1
  preStateS = (isVixOk * 1).rename('Pre State')
  stateS = ((preStateS.rolling(3).sum() >= 2) * 1).rename('State')
  dw[und] = cleanS(stateS, isMonthlyRebal=True)
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  #####
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['vixS'] = vixS
  d['vixRatioS'] = vixRatioS
  d['preStateS'] = preStateS
  d['stateS'] = stateS
  return d

def runVTS(yrStart, isSkipTitle=False):
  script = 'VTS'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runVTSCore(yrStart)
  #####
  st.header('Table')
  stWriteDf(ul.merge(d['dp'], d['vixS'], d['vixRatioS'].round(3), d['preStateS'], d['stateS'], how='inner').tail())
  #####
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

#####

def runVCACore(yrStart):
  und='VIXM'
  etc=ul.spl('SPY,VIX.INDX,VIX3M.INDX')
  dp, dw, dfDict, _ = btSetup([und]+etc,yrStart=yrStart-1)
  spyS = (dfDict['SPY']['Close']).rename('SPY')
  dp=applyDates(dp,spyS)
  dw=applyDates(dw,spyS)
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
  #####
  spyRatioS = (spyS / spyS.rolling(200).mean()).rename('SPY Ratio')
  ibsS = getIbsS(dfDict['SPY'])
  #####
  vixS = applyDates(dfDict['VIX.INDX']['Close'],spyS).rename('VIX')
  vix3MS = applyDates(dfDict['VIX3M.INDX']['Close'], spyS).rename('VIX3M')
  vixRatioS = (vixS / vixS.rolling(10).mean()).rename('VIX Ratio')
  hvS = (spyS.pct_change().rolling(10).std() * math.sqrt(252) * 100).rename('HV')
  eVRPS= (vixS-hvS).rename('eVRPS')
  eVRPS_pctl = eVRPS.rolling(252).rank(pct=True).rename('eVRPS Pctl')
  logRatioS = np.log(vixS / vix3MS)
  zScoreS = ((logRatioS - logRatioS.rolling(252).mean()) / logRatioS.rolling(252).std()).rename('ZScore')
  #####
  m= lambda s: applyDates(s,dw).ffill().fillna(0)
  w1 = m((spyRatioS < 1) & (ibsS > 0.75) & (vixRatioS > 1))
  w2 = m((eVRPS_pctl <= 0.25) & (vixRatioS > 1))
  w3 = m((zScoreS <= -1.5) & (vixRatioS > 1))
  dw[und] = cleanS((w1 + w2 + w3).clip(upper=1), isMonthlyRebal=False)
  dw=cleanS(dw,isMonthlyRebal=True)
  #####
  d=dict()
  d['dp']=dp
  d['dw']=dw
  d['SPY'] = spyS
  d['spyRatioS'] = spyRatioS
  d['ibsS'] = ibsS
  d['VIX'] = vixS
  d['vixRatioS'] = vixRatioS
  d['hvS']=hvS
  d['eVRPS'] = eVRPS
  d['eVRPS_pctl'] = eVRPS_pctl
  d['zScoreS'] = zScoreS
  return d

def runVCA(yrStart,isSkipTitle=False):
  script = 'VCA'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runVCACore(yrStart)
  st.header('Tables')
  tableS = ul.merge(d['dp'], d['SPY'], d['spyRatioS'].round(3), d['ibsS'].round(3),
                    d['VIX'], d['vixRatioS'].round(3), d['hvS'].round(2), d['eVRPS'].round(2), (d['eVRPS_pctl'] * 100).round(1), d['zScoreS'].round(3), how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runSCI(yrStart,isSkipTitle=False):
  script = 'SCI'
  if not isSkipTitle:
    st.header(script)
  ######
  dp, dw, dfDict, hv = btSetup(ul.spl('IEI,REM,GLD,GDXJ,XLE,OIH'),yrStart=yrStart-1)
  idx = dw.index[endpoints(dp)]
  dw.loc[idx, 'IEI'] = 0.8
  dw.loc[idx, 'REM'] = -0.4
  dw.loc[idx, 'GLD'] = 0.6
  dw.loc[idx, 'GDXJ'] = -0.2
  dw.loc[idx, 'XLE'] = 0.6
  dw.loc[idx, 'OIH'] = -0.4
  st.header('Prices')
  stWriteDf(dp.tail())
  st.header('Weights')
  dwTail(dw)
  bt(script, dp, dw, yrStart)

#####

def runAggregate(yrStart,strategies,weights,script,isBFill=False, isCorrs=False):
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
  if isBFill: dp=dp.bfill()
  dw = dp * np.nan
  pe = endpoints(dw)
  for i in range(len(weights)):
    dw.iloc[pe, i] = weights[i]
  #####
  # Backtest
  bt(script, dp, dw, yrStart)
  #####
  # Corrs
  if isCorrs:
    st.header('Corrs')
    stWriteDf(dp.pct_change().corr().round(3))
  #####
  # Recent performance
  st.header('Recent Performance')
  dp2 = dp.copy()
  dp2[script] = ul.cachePersist('r', script)
  dp2 = dp2[[script] + strategies]
  dp2 = (dp2 / dp2.iloc[-1]).tail(23) * 100
  dp2 = dp2.round(2)
  stWriteDf(dp2, isMaxHeight=True)
