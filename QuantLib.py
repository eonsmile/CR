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

def btSetup(tickers, hvN=32, yrStart=SHARED_DICT['yrStart'], applyDatesS=None):
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

def getCoreBetas(isZB=False):
  yrStart=pendulum.now().year-2
  iefS=getPriceHistory('IEF',yrStart=yrStart)['Close']
  znS = getYClose2Y('ZN=F')
  tnS = getYClose2Y('TN=F')

  d=dict()
  d['ZN_IEF']=getBeta(znS, iefS)
  d['TN_IEF']=getBeta(tnS, iefS)
  if isZB:
    tltS=getPriceHistory('TLT',yrStart=yrStart)['Close']
    zbS = getYClose2Y('ZB=F')
    ubS = getYClose2Y('UB=F')
    d['ZB_TLT'] = getBeta(zbS, tltS)
    d['UB_TLT'] = getBeta(ubS, tltS)
  return d

def getCoreWeightsDf():
  lastUpdateDict = ul.cachePersist('r','CR')['lastUpdateDict']
  fmt='DDMMMYY'
  dts = [pendulum.from_format(dt, fmt) for dt in lastUpdateDict.values()]
  lastUpdate = max(dts).format(fmt)
  l = list()
  ep = 1e-9
  #####
  emptyDict = {'SPY':0,'QQQ':0,'IEF':0,'GLD':0,'UUP':0}
  #####
  d = ul.cachePersist('r', 'CR')['TPPDict']
  tppDict=emptyDict.copy()
  for und in ul.spl('QQQ,IEF,GLD,UUP'):
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
  for und in ul.spl('SPY,QQQ,IEF,GLD,UUP'):
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
  if len(ticker) >= 3 and ticker.startswith('LU') and ticker[2].isdigit():
    ticker=f"{und}.EUFUND"
  elif '.' not in ticker:
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
  if und in ul.spl('CAOS,ORR,CCOM,COM,HARD,HGER,ASMF,CTA,DBMF,FFUT,HFMF,ISMF,KMLM,TFPN,IBIT'):
    if und=='CAOS':
      dtStart = '2023-3-31'
    elif und=='ORR':
      dtStart = '2025-1-31'
    elif und=='CCOM':
      with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        session = curl_cffi.Session(impersonate="chrome")
        df = yahooquery.Ticker(und, session=session).history()
      df.index = df.index.droplevel('symbol')
      df.index = pd.to_datetime(df.index.map(lambda x: pendulum.parse(str(x)).date()))
      df=df[['open','high','low','adjclose','volume']]
      df.columns=ul.spl('Open,High,Low,Close,Volume')
      dtStart = '2026-1-27'
    elif und=='HARD':
      dtStart = '2023-3-31'
    elif und=='HGER':
      dtStart = '2022-2-28'
    elif und=='ASMF':
      dtStart = '2024-5-31'
    elif und=='CTA':
      dtStart = '2022-3-31'
    elif und=='DBMF':
      dtStart = '2019-5-31'
    elif und=='FFUT':
      dtStart = '2025-6-30'
    elif und=='HFMF':
      dtStart = '2025-7-31'
    elif und=='ISMF':
      dtStart = '2025-3-31'
    elif und=='TFPN':
      dtStart = '2023-7-31'
    else:
      dtStart = None
    if dtStart is not None: df = df.loc[df.index >= dtStart]
    df = m(df, f"{und}.csv")
  elif und == 'BDRY':
    df = m(df, 'BDI.csv')
  elif und=='VIX1D.INDX':
    dtStart = '2023-4-24'
    df = df.loc[df.index>=dtStart]
  return df

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

def getYClose2Y(ticker):
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning)
    session = curl_cffi.Session(impersonate="chrome")
    df = yahooquery.Ticker(ticker,session=session).history(period='2y')
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
  volTgt = .23
  maxWgt = 2
  dp, dw, dfDict, hv = btSetup([und],yrStart=yrStart-1)
  #####
  df = dfDict[und]
  ibsS = getIbsS(df)
  isEntryS = ibsS < .1
  isExitS = df['Close'] > df['High'].shift(1)
  stateS = getStateS(isEntryS, isExitS, isCleaned=True, isMonthlyRebal=True)
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
  st.header('Tables')
  st.subheader(d['und'])
  df = d['dfDict'][d['und']]
  df2 = ul.merge(df['Close'].round(2), df['High'].round(2), df['Low'].round(2), d['ibsS'].round(3), how='inner')
  df2 = ul.merge(df2, d['stateS'].ffill(), how='inner')
  stWriteDf(df2.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runTPP(yrStart,multQ=1,multB=1,multG=1,multD=1,isSkipTitle=False):
  undQ = 'QQQ'
  undB = 'IEF'
  undG = 'GLD'
  undD = 'UUP'
  lookback = 32
  volTgt = .125
  maxWgt = 2
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
  st.header('Tables')
  tableS = ul.merge(d['dp'][d['und']],d['rsiS'].round(1),d['ibsS'].round(3),d['ratioS'].round(3),d['vixS'],d['vixRatioS'].round(3), d['stateS'].ffill(), how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

#############################################################################################


def runJJ5Core(yrStart):
  etc=ul.spl('BDRY,IGIB,VV,TQQQ')
  dp, dw, dfDict, hv = btSetup(ul.spl('QQQ,VIXY,SPY,GLD,EEM')+etc,yrStart=yrStart-1)
  #####
  rsiS_TQQQ = ta.rsi(dfDict['TQQQ']['Close'], length=10).rename('RSI')
  rsiS_BDRY = ta.rsi(dfDict['BDRY']['Close'], length=50)
  rsiS_IGIB = ta.rsi(dfDict['IGIB']['Close'], length=10)
  rsiS_VV = ta.rsi(dfDict['VV']['Close'], length=10)
  isOversoldS_TQQQ = applyDates(rsiS_TQQQ.rolling(2).max() < 25, dp).rename('OS') * 1.0
  isOverbotS_TQQQ = applyDates(rsiS_TQQQ.rolling(2).min() > 80, dp).rename('OB') * 1.0
  isCanaryS_BDRY=applyDates(rsiS_BDRY.rolling(2).min()>50,dp).rename('BDRY Canary')*1.0
  isCanaryS_IGIB=(applyDates(rsiS_IGIB>rsiS_VV,dp).rename('IGIB Canary'))*1.0
  #####
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
    hv = hv.drop(und2, axis=1)
  dw['QQQ'] = (isOversoldS_TQQQ == 1) * 3
  dw['VIXY'] = (isOverbotS_TQQQ == 1) * 1
  s = isOverbotS_TQQQ+isOversoldS_TQQQ
  dw['SPY'] = ((s == 0) & (isCanaryS_BDRY == 1)) * 2
  dw['GLD'] = ((s == 0) & (isCanaryS_BDRY==0) & (isCanaryS_IGIB==1))*1
  dw['EEM'] = ((s == 0) & (isCanaryS_BDRY==0) & (isCanaryS_IGIB==0))*-1
  dw=cleanS(dw*.65,isMonthlyRebal=True)
  #####
  dw.loc[dw.index.year < yrStart] = 0
  #####
  d=dict()
  d['dp']=dp
  d['dw']=dw
  d['rsiS_TQQQ'] = rsiS_TQQQ
  d['isOversoldS_TQQQ'] = isOversoldS_TQQQ
  d['isOverbotS_TQQQ'] = isOverbotS_TQQQ
  d['isCanaryS_BDRY']=isCanaryS_BDRY
  d['isCanaryS_IGIB']=isCanaryS_IGIB
  return d

def runJJ5(yrStart, isSkipTitle=False):
  script = 'JJ5'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runJJ5Core(yrStart)
  st.header('Tables')
  tableS = ul.merge(d['dp'],d['rsiS_TQQQ'].round(1),d['isOversoldS_TQQQ'],d['isOverbotS_TQQQ'],d['isCanaryS_BDRY'],d['isCanaryS_IGIB'], how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runJJ3Core(yrStart):
  def getSMA12Mom(s):
    sum = 0
    for i in range(13):
      sum += s.shift(i * 21)
    return 13 * s / sum - 1
  #####
  etc=ul.spl('TQQQ,SPHB,SPLV')
  dp, dw, dfDict, hv = btSetup(ul.spl('QQQ,GLD,VIXY')+etc,yrStart=yrStart-1)
  #####
  rsi5S_TQQQ = ta.rsi(dfDict['TQQQ']['Close'], length=5)
  rsi9S_TQQQ = ta.rsi(dfDict['TQQQ']['Close'], length=9)
  rsi14S_TQQQ = ta.rsi(dfDict['TQQQ']['Close'], length=14)
  isOversold5S_TQQQ= applyDates(rsi5S_TQQQ.rolling(2).max()<30,dp)*1.0
  isOversold9S_TQQQ= applyDates(rsi9S_TQQQ.rolling(2).max()<30,dp)*1.0
  isOversold14S_TQQQ= applyDates(rsi14S_TQQQ.rolling(2).max()<30,dp)*1.0
  isOversoldCountS_TQQQ = (isOversold5S_TQQQ+isOversold9S_TQQQ+isOversold14S_TQQQ).rename('OS Count')
  isOverbot5S_TQQQ = applyDates(rsi5S_TQQQ>80,dp)*1.0
  isOverbot9S_TQQQ = applyDates(rsi9S_TQQQ>80,dp)*1.0
  isOverbot14S_TQQQ = applyDates(rsi14S_TQQQ>80,dp)*1.0
  isOverbotCountS_TQQQ = (isOverbot5S_TQQQ + isOverbot9S_TQQQ + isOverbot14S_TQQQ).rename('OB Count')
  dw['VIXY']=(isOverbotCountS_TQQQ>=2)*0.5+(isOverbotCountS_TQQQ==3)*0.5
  dw['QQQ']=((isOversoldCountS_TQQQ>=2)*0.5+(isOversoldCountS_TQQQ==3)*0.5)*3
  ratioS=dfDict['SPHB']['Close']/dfDict['SPLV']['Close']
  momS=applyDates(getSMA12Mom(ratioS),dp).rename('SPHB/SPLV Mom')
  isRegularS = (isOverbotCountS_TQQQ <2) & (isOversoldCountS_TQQQ <2)
  dw['QQQ']+=(isRegularS & (momS.rolling(2).min()>0))*2
  dw['GLD']=(isRegularS & (momS.rolling(2).min()<=0))*1
  #####
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
    hv = hv.drop(und2, axis=1)
  dw=cleanS(dw*.65,isMonthlyRebal=True)
  #####
  dw.loc[dw.index.year < yrStart] = 0
  #####
  d=dict()
  d['dp']=dp
  d['dw']=dw
  d['isOversoldCountS_TQQQ'] = isOversoldCountS_TQQQ
  d['isOverbotCountS_TQQQ'] = isOverbotCountS_TQQQ
  d['momS'] = momS
  return d

def runJJ3(yrStart, isSkipTitle=False):
  script = 'JJ3'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runJJ3Core(yrStart)
  st.header('Tables')
  tableS = ul.merge(d['dp'],d['isOversoldCountS_TQQQ'],d['isOverbotCountS_TQQQ'],d['momS'].round(3), how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runJJ2Core(yrStart):
  etc=ul.spl('WOOD,XLU,BDRY')
  dp, dw, dfDict, hv = btSetup(ul.spl('SPY,GLD')+etc,yrStart=yrStart-1)
  #####
  isCanaryS_WOOD = applyDates((dfDict['WOOD']['Close'].pct_change(200)>dfDict['GLD']['Close'].pct_change(200)).rolling(5).sum()==5,dp).rename('WOOD Canary')*1.0
  isCanaryS_XLU = applyDates(ta.rsi(dfDict['SPY']['Close'], length=20)>ta.rsi(dfDict['XLU']['Close'], length=20).rename('RSI'),dp).rename('XLU Canary')*1.0
  cS_BDRY=dfDict['BDRY']['Close']
  icb1 = (cS_BDRY>EMA(cS_BDRY,250)).rolling(3).sum()==3
  icb2 = (cS_BDRY.pct_change(60)>.1).rolling(2).sum()==2
  isCanaryS_BDRY = applyDates(icb1 | icb2,dp).rename('BDRY Canary')*1.0
  nCanaries = isCanaryS_WOOD+isCanaryS_XLU+isCanaryS_BDRY
  aroonUpS_SPY = applyDates(ta.aroon(dfDict['SPY']['High'], dfDict['SPY']['Low'], length=200)['AROONU_200'], dp).rename('SPY AroonUp')
  #####
  dw['SPY'] = nCanaries
  mask=(nCanaries==3) & (aroonUpS_SPY>=95)
  dw.loc[mask, 'SPY'] -= 1
  dw['GLD']=(nCanaries==0)*1
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
    hv = hv.drop(und2, axis=1)
  dw = cleanS(dw*0.7, isMonthlyRebal=True)
  #####
  dw.loc[dw.index.year < yrStart] = 0
  #####
  d=dict()
  d['dp']=dp
  d['dw']=dw
  d['isCanaryS_WOOD']=isCanaryS_WOOD
  d['isCanaryS_XLU']=isCanaryS_XLU
  d['isCanaryS_BDRY']=isCanaryS_BDRY
  d['aroonUpS_SPY']=aroonUpS_SPY
  return d

def runJJ2(yrStart, isSkipTitle=False):
  script = 'JJ2'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runJJ2Core(yrStart)
  st.header('Tables')
  tableS = ul.merge(d['dp'], d['isCanaryS_WOOD'], d['isCanaryS_XLU'], d['isCanaryS_BDRY'], d['aroonUpS_SPY'], how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

#####

def runBCSCore(yrStart):
  und='SPY'
  dp, dw, dfDict, hv = btSetup([und],yrStart=yrStart-1)
  #####
  btcS = getPriceHistoryCrypto('BTC', yrStart - 1)['Close'].rename('BTC')
  ratio50S = (btcS/btcS.rolling(50).mean()).rename('Ratio 50D')
  ratio300S = (btcS/btcS.rolling(300).mean()).rename('Ratio 300D')
  dw[und]=cleanS(applyDates((ratio50S>1)&(ratio300S>1),dp),isMonthlyRebal=True)
  #####
  d=dict()
  d['dp']=dp
  d['dw']=dw
  d['btcS'] = btcS
  d['ratio50S'] = ratio50S
  d['ratio300S']=ratio300S
  return d

def runBCS(yrStart,isSkipTitle=False):
  script = 'BCS'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runBCSCore(yrStart)
  st.header('Tables')
  tableS = ul.merge(d['dp'],d['btcS'],d['ratio50S'].round(3),d['ratio300S'].round(3), how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runHYSCore(yrStart):
  volTgt = .15
  maxWgt = 2
  dp, dw, dfDict, hv = btSetup(ul.spl('SPY,BTAL,HYG'),yrStart=yrStart-1)
  #####
  hygS = dfDict['HYG']['Close'].rename('HYG')
  ratioS = (hygS/EMA(hygS,100)).rename('Ratio')
  stateS = applyDates(ratioS>1,dp)*1
  #####
  dp = dp.drop('HYG', axis=1)
  dw = dw.drop('HYG', axis=1)
  hv = hv.drop('HYG', axis=1)
  #####
  dw['SPY'] = stateS
  dw['BTAL'] = 1 - stateS
  dw=cleanS(dw,isMonthlyRebal=True)
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  dw.loc[dw.index.year < yrStart] = 0
  #####
  d=dict()
  d['dp']=dp
  d['dw']=dw
  d['hygS'] = hygS
  d['ratioS']=ratioS
  return d

def runHYS(yrStart,isSkipTitle=False):
  script = 'HYS'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runHYSCore(yrStart)
  st.header('Tables')
  tableS = ul.merge(d['dp'],d['hygS'],d['ratioS'], how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

#####

def runQS12Core(yrStart):
  undG = 'GLD'
  undB = 'TLT'
  volTgt = .16
  maxWgt = 2
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


def runBSSCore(yrStart):
  und = 'TLT'
  etc = ul.spl('SPY')
  dp, dw, dfDict, _ = btSetup([und] + etc, yrStart=yrStart - 1)
  cS = dp['SPY']
  pe = endpoints(cS)[:-1]
  anchorS = cS.copy()
  anchorS[:] = np.nan
  anchorS.iloc[pe + 1] = cS.iloc[pe]
  anchorS = anchorS.ffill()
  mtdS = (cS / anchorS - 1).rename('SPY MTD')
  isOkS = (mtdS > 0) * 1
  #####
  pe = endpoints(dp, -6)[:-1]
  selection = dw.index[pe]
  dw.loc[selection, und] = applyDates(isOkS, dw)[selection]
  dt = getNYSEMonthEnd(-6)
  msg=f"ME-6: {dt:%Y-%m-%d}"
  if dt in dw.index:
    dw.loc[dt, und] = isOkS[dt]
  #####
  pe = endpoints(dp)[:-1]
  selection = dw.index[pe]
  dw.loc[selection, und] = 0
  dt = getNYSEMonthEnd()
  if dt in dw.index:
    dw.loc[dt] = 0
  #####
  dw[und] = cleanS(dw[und], isMonthlyRebal=False)
  dw.loc[dw.index.year < yrStart] = 0
  #####
  dp2 = dp.copy()
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
  #####
  d=dict()
  d['dp']=dp
  d['dp2']=dp2
  d['dw']=dw
  d['mtdS'] = mtdS
  d['msg']=msg
  return d

def runBSS(yrStart, isSkipTitle=False):
  script = 'BSS'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runBSSCore(yrStart)
  st.write(d['msg'])
  st.header('Tables')
  tableS = ul.merge(d['dp2'], d['mtdS'].round(3),how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runVRSCore(yrStart):
  lw = .5
  sw = lw/2
  #       Calmar: 4.39          MAR: 4.30          Sharpe: 3.01          Cagr: 36.5%          MaxDD: 8.5%
  #####
  und = 'VIXY'
  etc = ul.spl('VIX1D.INDX,VIX.INDX,VIX3M.INDX')
  dp, dw, dfDict, _ = btSetup([und], yrStart=yrStart - 1)
  dp2, dw2, dfDict2, _ = btSetup(etc, yrStart=yrStart - 1)
  vix1DS = applyDates(dfDict2['VIX1D.INDX']['Close'], dp)
  vixS = applyDates(dfDict2['VIX.INDX']['Close'], dp)
  vix3MS = applyDates(dfDict2['VIX3M.INDX']['Close'], dp)
  dw[und] = ((vix1DS <= 10) * lw - (vix1DS >= 15) * sw) * (vixS <= vix3MS)
  #####
  dw = cleanS(dw, isMonthlyRebal=False)
  dw.loc[dw.index.year < yrStart] = 0
  d = dict()
  d['dp'] = dp
  d['dp2'] = dp2
  d['dw'] = dw
  return d

def runVRS(yrStart,isSkipTitle=False):
  script = 'VRS'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runVRSCore(yrStart)
  st.header('Prices')
  dwTail(ul.merge(d['dp'],d['dp2'],how='inner'))
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

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

def runCore(yrStart):
  runTPP(yrStart)
  st.divider()
  runRSS(yrStart)
  st.divider()
  runIBS(yrStart)
  st.divider()
  strategies = ul.spl('TPP,RSS,IBS')
  weights = [1 / 2, 1 / 4, 1 / 4]
  script = 'Core'
  runAggregate(yrStart, strategies, weights, script, isCorrs=True)
