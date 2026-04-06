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
import pandas_market_calendars
import yahooquery
import curl_cffi
import warnings
import pandas_ta_classic as ta
from sklearn.linear_model import LinearRegression

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

def getNYSEMonthEnd(offset=0):
  tz = 'America/New_York'
  now = pendulum.now(tz)

  monthStart = now.start_of('month').to_date_string()
  monthEnd = now.end_of('month').to_date_string()

  nyseCalendar = pandas_market_calendars.get_calendar('NYSE')
  schedule = nyseCalendar.schedule(start_date=monthStart, end_date=monthEnd)

  sessions = schedule.index
  if len(sessions) == 0:
    raise ValueError(f'No NYSE sessions found for {now.format("YYYY-MM")}')

  targetIdx = (len(sessions) - 1) + offset
  if targetIdx < 0 or targetIdx >= len(sessions):
    raise ValueError(
      f'offset {offset} is out of range for {now.format("YYYY-MM")} '
      f'(valid offsets: {-len(sessions) + 1}..0)'
    )

  return pd.Timestamp(sessions[targetIdx]).normalize()

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
    totalWeight = tppDict[und]*.5+rssDict[und]*.25+ibsDict[und]*.25
    l.append([dts[i], und, totalWeight, tppDict[und], rssDict[und], ibsDict[und]])
    i += 1
  df = pd.DataFrame(l)
  df.columns = ul.spl('Last Update,ETF,Total Weight,TPP (1/2),RSS (1/4),IBS (1/4)')
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

def getIbsS(df,n=1):
  if n==1:
    ibsS = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
  else:
    lS=df['Low'].rolling(n).min()
    hS=df['High'].rolling(n).max()
    ibsS = (df['Close'] - lS) / (hS - lS)
  ibsS.name = 'IBS'
  return ibsS

def getPriceHistory(und, yrStart=SHARED_DICT['yrStart']):
  dtStart=str(yrStart)+ '-1-1'
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
  def m(df,fn):
    df2 = pd.read_csv(f"data/{fn}", index_col=0, parse_dates=True, date_format='%m/%d/%Y')
    for col in ['Open', 'High', 'Low', 'Volume']:
      df2[col] = df2['Close'] * (0 if col == 'Volume' else 1)
    return extend(df, df2)
  #####
  if und in ul.spl('GDXJ,EUDF.XETRA,IPRE.XETRA,'
                   'COM,DBMF,INFL,PFMN.TO,'
                   'COPX,ENCO.LSE,DFNS.LSE,GCOW,JEGA.LSE,NATO.LSE,ORR,PFIX,RARE.LSE,REMX,ROLL.LSE,TAIL,WCOA.LSE,'
                   'IBIT'):
    if und=='GDXJ':
      dtStart='2009-11-30'
    elif und == 'EUDF.XETRA':
      dtStart = '2025-3-31'
    elif und == 'IPRE.XETRA':
      dtStart = '2018-12-28'
    #####
    # COM
    elif und=='DBMF':
      dtStart = '2019-5-31'
    elif und=='INFL':
      dtStart = '2021-1-29'
    elif und == 'PFMN.TO':
      dtStart = '2019-7-31'
    #####
    elif und=='COPX':
      dtStart='2010-4-30'
    elif und=='ENCO.LSE':
      dtStart='2021-8-31'
    elif und=='DFNS.LSE':
      dtStart='2023-4-28'
    elif und=='GCOW':
      dtStart='2016-2-29'
    elif und == 'JEGA.LSE':
      dtStart = '2023-12-29'
    elif und == 'NATO.LSE':
      dtStart = '2023-7-31'
    elif und=='ORR':
      dtStart = '2025-1-31'
    elif und=='PFIX':
      dtStart='2021-5-28'
    elif und=='RARE.LSE':
      dtStart='2024-4-30'
    elif und=='REMX':
      dtStart='2010-10-29'
    elif und=='ROLL.LSE':
      dtStart = '2020-12-29'
    elif und=='TAIL':
      dtStart = '2017-4-28'
    elif und=='WCOA.LSE':
      dtStart = '2025-9-30'
    #####
    elif und=='IBIT':
      dtStart = '2024-1-11'
    else:
      dtStart = None
    if dtStart is not None: df = df.loc[df.index >= dtStart]
    df = m(df, f"{und}.csv")
  elif und == 'DFND.SW':
    df = m(df, 'ITA.csv')
  elif und == 'BDRY':
    df = m(df, 'BDI.csv')
  elif und=='VIX1D.INDX':
    dtStart = '2023-4-24'
    df = df.loc[df.index>=dtStart]
  return df

  #df2 = getPriceHistory('ITA', yrStart=yrStart)
  #df2[['Close']].to_csv('tmp.csv', index_label='Date', date_format='%#m/%#d/%Y')

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

def getPriceHistoryFred(id, yrStart=SHARED_DICT['yrStart']):
  dtStart = f"{yrStart}-01-01"
  url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd={dtStart}"
  df = pd.read_csv(url, parse_dates=['observation_date'], index_col='observation_date')
  df.columns = [id]
  df = df.replace('.', np.nan).astype(float).dropna()
  df.index.name = None
  return df[id]

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

def getYClose(ticker, period=2):
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning)
    session = curl_cffi.Session(impersonate="chrome")
    df = yahooquery.Ticker(ticker,session=session).history(period=f"{period}y")
  df.index = df.index.droplevel('symbol')
  df.index = pd.to_datetime(df.index.map(lambda x: pendulum.parse(str(x)).date()))
  return df['adjclose'].rename(ticker)

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
def runIBSCore(yrStart, mult=1):
  und = 'QQQ'
  volTgt = .255
  maxWgt = 2
  dp, dw, dfDict, hv = btSetup([und],yrStart=yrStart-1)
  #####
  df = dfDict[und]
  ibsS = getIbsS(df)
  isEntryS = ibsS < .1
  isExitS = (ibsS > .9) | (df['Close'] > df['High'].shift(1))
  stateS = getStateS_timestop(isEntryS, isExitS, 10, isCleaned=True, isMonthlyRebal=True)
  dw[und] = stateS*mult
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  dwAllOrNone(dw)
  d=dict()
  d['und']=und
  d['dp']=dp
  d['dw']=dw
  d['dfDict']=dfDict
  d['ibsS']=ibsS
  d['stateS']=stateS
  return d

def runIBS(yrStart,mult=1, isSkipTitle=False):
  script = 'IBS'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runIBSCore(yrStart,mult=mult)
  st.header('Table')
  st.subheader(d['und'])
  df = d['dfDict'][d['und']]
  df2 = ul.merge(df['Close'].round(2), df['High'].round(2), df['Low'].round(2), d['ibsS'].round(3), how='inner')
  df2 = ul.merge(df2, d['stateS'].ffill(), how='inner')
  stWriteDf(df2.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runRSSCore(yrStart):
  und='SPY'
  volTgt = .22
  maxWgt = 2
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
  dw[und] = stateS
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  dw.loc[dw.index.year < yrStart] = 0
  d=dict()
  d['und']=und
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
  tableS = ul.merge(d['dp'][d['und']],d['rsiS'].round(1),d['ibsS'].round(3),d['ratioS'].round(3),d['vixS'],d['vixRatioS'].round(3), d['stateS'].ffill(), how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)


def runTPP(yrStart,multQ=1,multB=1,multG=1,multD=1,isSkipTitle=False):
  undQ = 'QQQ'
  undB = 'IEI'
  undG = 'GLD'
  undD = 'UUP'
  lookback = 32
  volTgt = .125
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
  volTgt = .165
  maxWgt = 1.5
  etc=ul.spl('HYG,SPHB,SPLV,WOOD,IGIB,VV,TIP')
  dp, dw, dfDict, hv = btSetup(ul.spl('SPY,GLD,UUP')+etc,yrStart=yrStart-1)
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
    hv = hv.drop(und2, axis=1)
  #####

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

  # 3. IGIB
  rsiS_IGIB = ta.rsi(applyDates(dfDict['IGIB']['Close'],dp), length=10)
  rsiS_VV = ta.rsi(applyDates(dfDict['VV']['Close'],dp), length=10)
  isCanaryS_IGIB = ((rsiS_IGIB > rsiS_VV).rename('IGIB')) * 1.0

  # 4. SPHB/LV momentum
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

  # 5. TIP
  cS_TIP = applyDates(dfDict['TIP']['Close'],dp)
  momS_TIP = (cS_TIP.pct_change(21) + cS_TIP.pct_change(63) + cS_TIP.pct_change(126) + cS_TIP.pct_change(252)) / 4
  isCanaryS_TIP = (momS_TIP > 0).rename('TIP') * 1.0

  # Voting
  voteDf = pd.DataFrame({
    'BTC': isCanaryS_BTC,
    'HYG': isCanaryS_HYG,
    'IGIB': isCanaryS_IGIB,
    'SPHB_LV': isCanaryS_SPHB_LV,
    'TIP': isCanaryS_TIP,
  })
  voteCountS = voteDf.sum(axis=1).rename('Votes')

  ##################
  # UUP/GLD canaries
  ##################
  gldS = applyDates(dfDict['GLD']['Close'],dp)
  woodS = applyDates(dfDict['WOOD']['Close'],dp)
  ratio150S_BTC = (btcS / btcS.rolling(150).mean()).rename('BTC Ratio 150D')
  ratio200S_SPHB_LV = (sphb_lv_ratio / sphb_lv_ratio.rolling(252).mean()).rename('SPHB_LV 252D')
  mom200S_GLD_WOOD = (gldS.pct_change(200)-woodS.pct_change(200)).rename('GLD_WOOD 200D')
  #####
  m = lambda n: applyDates(n, dw) * 1
  dw['SPY'] = (voteCountS >= 2) * (voteCountS / 5)
  dw['GLD']=(m(ratio200S_SPHB_LV<1)+m(mom200S_GLD_WOOD>0))/2
  dw['UUP'] = m(ratio150S_BTC < 1)
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
  d['isCanaryS_IGIB'] = isCanaryS_IGIB
  d['isCanaryS_SPHB_LV'] = isCanaryS_SPHB_LV
  d['isCanaryS_TIP'] = isCanaryS_TIP
  #####
  d['ratio200S_SPHB_LV']=ratio200S_SPHB_LV
  d['mom200S_GLD_WOOD']=mom200S_GLD_WOOD
  d['ratio150S_BTC'] = ratio150S_BTC
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
    d['isCanaryS_IGIB'], d['isCanaryS_SPHB_LV'], d['isCanaryS_TIP'],
    d['voteCountS'], how='inner').tail())
  st.header('GLD/UUP Ratios')
  stWriteDf(ul.merge(applyDates(d['ratio200S_SPHB_LV'],dp).round(3), applyDates(d['mom200S_GLD_WOOD'], dp).round(3), applyDates(d['ratio150S_BTC'], dp).round(3), how='inner').tail())
  st.header('States')
  stWriteDf(d['stateDf'].tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runBTSCore(yrStart):
  HALVINGS = pd.to_datetime(['2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20'])
  #####
  volTgt = 0.16
  maxWgt = 1
  cS = getPriceHistoryCrypto('BTC', yrStart=yrStart)['Close']
  ratioS = (cS/cS.rolling(50).mean()).rename('Ratio')
  nDays_off = 450
  nDays_on = 900
  isSeasonS = pd.Series(1, index=cS.index).rename('Season?')
  for hd in HALVINGS:
    isSeasonS[(cS.index >= hd + pd.Timedelta(days=nDays_off)) &
         (cS.index < hd + pd.Timedelta(days=nDays_on))] = 0
  dw = ((ratioS>1) & isSeasonS).rename('BTC').to_frame()
  dp = cS.rename('BTC').to_frame()
  hv = getHV(dp, af=365)
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  dw = cleanS(dw, isMonthlyRebal=True)
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['isSeasonS'] = isSeasonS
  d['ratioS'] = ratioS
  return d

def runBTS(yrStart, isSkipTitle=False):
  script = 'BTS'
  if not isSkipTitle:
    st.header(script)
  d = runBTSCore(yrStart)
  st.header('Table')
  tableS = ul.merge(d['dp'], d['isSeasonS'], d['ratioS'].round(3), how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)


def runGEOCore(yrStart):
  volTgt = .05
  maxWgt = 1
  etc = ul.spl('U-UN.TO,ITA,XME')
  dp, dw, dfDict, hv = btSetup(ul.spl('CCO.TO,NATO.LSE,COPX,REMX') + etc, yrStart=yrStart-1)
  dp2 = dp.copy()
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
    hv = hv.drop(und2, axis=1)
  #####
  # CCO signal: U-UN.TO monthly ROC
  uunS = dfDict['U-UN.TO']['Close']
  rocS_UUN = (uunS.iloc[endpoints(uunS)].pct_change()).rename('UUN ROC 1M')

  # NATO signal: ITA 10-month SMA
  itaS = dfDict['ITA']['Close']
  itaMS = itaS.iloc[endpoints(itaS)]
  ratio10S_ITA = (itaMS / itaMS.rolling(10).mean()).rename('ITA Ratio 10M')

  # COPX signal: XME 6-month SMA
  xmeS = dfDict['XME']['Close']
  xmeMS = xmeS.iloc[endpoints(xmeS)]
  ratio6S_XME = (xmeMS / xmeMS.rolling(6).mean()).rename('XME Ratio 6M')

  # REMX signal: REMX 10-month SMA
  remxS = dfDict['REMX']['Close']
  remxMS = remxS.iloc[endpoints(remxS)]
  ratio10S_REMX = (remxMS / remxMS.rolling(10).mean()).rename('REMX Ratio 10M')

  #####
  dw['CCO.TO'] = applyDates(rocS_UUN > 0, dw)
  dw['NATO.LSE'] = applyDates(ratio10S_ITA > 1, dw)
  dw['COPX'] = applyDates(ratio6S_XME > 1, dw)
  dw['REMX'] = applyDates(ratio10S_REMX > 1, dw)
  dw=cleanS(dw, isMonthlyRebal=True)
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  #####
  d = dict()
  d['dp'] = dp
  d['dp2'] = dp2
  d['dw'] = dw
  d['rocS_UUN'] = rocS_UUN
  d['ratio10S_ITA'] = ratio10S_ITA
  d['ratio6S_XME'] = ratio6S_XME
  d['ratio10S_REMX'] = ratio10S_REMX
  return d

def runGEO(yrStart, isSkipTitle=False):
  script = 'GEO'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runGEOCore(yrStart)
  st.header('Prices')
  stWriteDf(d['dp2'].tail())
  st.header('Ratios')
  stWriteDf(ul.merge(d['rocS_UUN'].round(3), d['ratio10S_ITA'].round(3), d['ratio6S_XME'].round(3), d['ratio10S_REMX'].round(3), how='inner').tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runQS12Core(yrStart):
  undG = 'GLD'
  undB = 'TLT'
  volTgt = .16
  maxWgt = 1
  tickers = [undG,undB]
  dp, dw, dfDict, hv = btSetup(tickers, yrStart=yrStart-1)
  hS = dfDict[undG]['High']
  cS = dfDict[undG]['Close']
  cSB = dfDict[undB]['Close']
  #####
  cond1S = cS > hS.shift(1).rolling(3).max()
  cond2S = cSB > cSB.shift()
  cond3S = (cS * 0).astype(int)
  cond3S.loc[cond3S.index.weekday != 3] = 1
  isEntryS = (cond1S & cond2S & cond3S)*1
  isExitS = (cS>hS.shift())*1
  isExitS.loc[isEntryS == 1] = 0
  stateS = getStateS(isEntryS, isExitS, isCleaned=True, isMonthlyRebal=True)
  #####
  # Summary
  dp=dp.drop(undB,axis=1)
  dw=dw.drop(undB,axis=1)
  hv=hv.drop(undB,axis=1)
  dw[undG] = stateS
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  dw.loc[dw.index.year < yrStart] = 0
  #####
  d=dict()
  d['dp'] = dp
  d['dw'] = dw
  d['dfDict'] = dfDict
  #####
  d['cS'] = cS
  d['hS'] = hS
  d['cSB'] = cSB
  d['stateS']=stateS
  return d

def runQS12(yrStart, isSkipTitle=False):
  script = 'QS12'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runQS12Core(yrStart)
  st.header('Table')
  tableS = ul.merge(d['cS'].round(2), d['hS'].round(2), d['cSB'].rename('Close (TLT)').round(2),d['stateS'].ffill(), how='inner')
  stWriteDf(tableS.tail())
  #####
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

#####

def runVCACore(yrStart):
  und='VIXM'
  etc=ul.spl('SPY,VIX.INDX,VIX1D.INDX')
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
  vix1DS = applyDates(dfDict['VIX1D.INDX']['Close'], spyS).rename('VIX1D')
  vixRatioS = (vixS / vixS.rolling(10).mean()).rename('VIX Ratio')
  hvS = (spyS.pct_change().rolling(10).std() * math.sqrt(252) * 100).rename('HV')
  eVRPS= (vixS-hvS).rename('eVRPS')
  eVRPS_pctl = eVRPS.rolling(252).rank(pct=True).rename('eVRPS Pctl')
  #####
  m= lambda s: applyDates(s,dw).ffill().fillna(0)
  w1 = m((spyRatioS < 1) & (ibsS > 0.75) & (vixRatioS > 1))
  w2 = m((eVRPS_pctl <= 0.25) & (vixRatioS > 1))
  w3=m(vix1DS <= 10)
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
  d['VIX1D'] = vix1DS
  return d

def runVCA(yrStart,isSkipTitle=False):
  script = 'VCA'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runVCACore(yrStart)
  st.header('Tables')
  tableS = ul.merge(d['dp'],d['SPY'],d['spyRatioS'].round(3),d['ibsS'].round(3),
                    d['VIX'],d['vixRatioS'].round(3),d['hvS'].round(2),d['eVRPS'].round(2),(d['eVRPS_pctl']*100).round(1),d['VIX1D'], how='inner')
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

def runSCI2(yrStart,isSkipTitle=False):
  script = 'SCI2'
  if not isSkipTitle:
    st.header(script)
  ######
  volTgt = .08
  maxWgt = 1
  dp, _, dfDict, _ = btSetup(ul.spl('IWM,XLRE,KBWD,JETS,IPRE.XETRA,KRE,'
                               'XLV,XLU,MOAT,DFND.SW,EUDF.XETRA,DB1.XETRA'),yrStart=yrStart-1)
  dp = applyDates(dp,dfDict['XLRE'])
  dp['S68.SI'] = applyDates(getYClose('S68.SI', period=20), dp)
  dw=dp.copy()
  dw[:]=np.nan
  hv=getHV(dp)
  #####
  idx = dw.index[endpoints(dw)]
  dw.loc[idx, 'IWM'] = -1
  dw.loc[idx, 'XLRE'] = -1
  dw.loc[idx, 'KBWD'] = -1
  dw.loc[idx, 'JETS'] = -1
  dw.loc[idx, 'IPRE.XETRA'] = -1
  dw.loc[idx, 'KRE'] = -1
  #####
  dw.loc[idx, 'XLV'] = 1
  dw.loc[idx, 'XLU'] = 1
  dw.loc[idx, 'MOAT'] = 1
  dw.loc[idx, 'DFND.SW'] = 1
  dw.loc[idx, 'EUDF.XETRA'] = 1 # WDEF in IBKR
  dw.loc[idx, 'DB1.XETRA'] = .5
  dw.loc[idx, 'S68.SI'] = .5
  dw = (dw * volTgt / hv).clip(-maxWgt, maxWgt)
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
