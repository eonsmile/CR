###############
# Quant Library
###############
import UtilLib as ul
import streamlit as st
import numpy as np
import pandas as pd
import requests
import math
import quandl
import pendulum
import pykalman
import yfinance as yf
import pandas_market_calendars
from sklearn.linear_model import LinearRegression
import pandas_ta

###########
# Constants
###########
quandl.ApiConfig.api_key = st.secrets['quandl_api_key']
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
MKT_CLOSE_HOUR=4

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
  nYears = (ecS.index[-1] - ecS.index[0]).days / 365
  cagr = math.pow(ecS.iloc[-1] / ecS.iloc[0], 1 / nYears) - 1
  dd = ecS / ecS.cummax() - 1
  upi = cagr / np.sqrt(np.power(dd, 2).mean())
  maxDD = -min(dd)
  vol = ((np.log(ecS / ecS.shift(1)) ** 2).mean()) ** 0.5 * (252 ** 0.5)

  m=lambda label,z: f"{label}: <font color='red'>{z}</font>"
  sep='&nbsp;'*10
  st.markdown(sep.join([
    m('&nbsp;'*3+'UPI', f"{upi:.2f}"),
    m('Sharpe', f"{cagr / vol:.2f}"),
    m('Cagr', f"{cagr:.1%}"),
    m('MaxDD', f"{maxDD:.1%}")
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
  ul.stWriteDf(dw.mask(dw.abs() == 0.0, 0.0).dropna().tail(n).round(3))

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

# https://quantstrattrader.wordpress.com/author/ikfuntech/
def endpoints(df, on='ME', offset=0):
  ep_dates = pd.Series(df.index, index=df.index).resample(on).max()
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

def getIsSeasonalS(s,mEntry,dEntry,mExit,dExit):
  s=s.copy()
  s[:]=np.nan
  for y in range(s.index.year[0],s.index.year[-1]+1):
    dtEntry=getBizDayPriorNYSE(y,mEntry,dEntry)
    dtExit=getBizDayNYSE(y, mExit, dExit, isAdjForward=False)
    if dtEntry in s.index:
      s.loc[dtEntry]=1
    elif dtEntry < s.index[-1]:
      print(f"getSeasonalS: {dtEntry:%Y-%m-%d} (entry) is missing!")
    if dtExit in s.index:
      s.loc[dtExit]=0
    elif dtExit < s.index[-1]:
      print(f"getSeasonalS: {dtExit:%Y-%m-%d} (exit) is missing!")
  return s.ffill().fillna(0).rename('Seasonal?')

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
    if isMonthlyRebal:
      pe=endpoints(s, 'ME')
      s.iloc[pe]=s.ffill().iloc[pe]
  else:
    for i in range(1, len(s)):
      if s.iloc[i]==tmp.iloc[i]:
        s.iloc[i]=np.nan
    if isMonthlyRebal:
      pe=endpoints(s, 'ME')
      s[pe]=s.ffill()[pe]
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
  else: # Quandl
    df = quandl.get_table('QUOTEMEDIA/PRICES', ticker=und, paginate=True, date={'gte': dtStart})
    df = df[ul.spl('date,adj_open,adj_high,adj_low,adj_close,adj_volume')]
    df = df[df['adj_volume'] != 0]  # Correction for erroneous zero volume days
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

def getYFinanceS(ticker):
  from_date = f"{START_YEAR_DICT['YFinance']}-01-01"
  to_date = pendulum.today().format('YYYY-MM-DD')
  return yf.download(ticker, start=from_date, end=to_date)['Adj Close'].rename(ticker)

#############################################################################################

#########
# Scripts
#########
def runIBSCore():
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
    stateS = getStateS(isEntryS, isExitS, isCleaned=True, isMonthlyRebal=False)
    return ibsS, stateS
  #####
  undE = 'SPY'
  undQ = 'QQQ'
  undB = 'TLT'
  volTgt = .16
  maxWgt = 1
  dp, dw, dfDict, hv = btSetup([undE, undQ, undB])
  #####
  isMondayS = dfDict[undE]['Close'].rename('Monday?') * 0
  isMondayS[isMondayS.index.weekday == 0] = 1
  ibsSE, stateSE = m(undE, dfDict, isMondayS=isMondayS)
  ibsSQ, stateSQ = m(undQ, dfDict)
  ibsSB, stateSB = m(undB, dfDict)
  dw[undE] = stateSE
  dw[undQ] = stateSQ
  dw[undB] = stateSB
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

def runIBS(yrStart=START_YEAR_DICT['IBS']):
  def m(d, und, ibsS, stateS, isMondayS=None):
    df=d['dfDict'][und]
    st.subheader(und)
    df2 = ul.merge(df['Close'].round(2), df['High'].round(2), df['Low'].round(2), ibsS.round(3), how='inner')
    if isMondayS is not None: df2 = ul.merge(df2, isMondayS, how='inner')
    df2 = ul.merge(df2, stateS.ffill(), how='inner')
    ul.stWriteDf(df2.tail())
  #####
  script = 'IBS'
  st.header(script)
  d=runIBSCore()
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
  for i in endpoints(rDf, 'ME'):
    origin = i - lookback + 1
    if origin >= 0:
      prS = rDf.iloc[origin:(i + 1)].multiply(wDf.iloc[i], axis=1).sum(axis=1)
      pHv = ((prS ** 2).mean()) ** .5 * (252 ** .5)
      dw.iloc[i] = wDf.iloc[i] * volTgt / pHv
  dw.clip(0, maxWgt, inplace=True)
  st.header('Prices')
  ul.stWriteDf(dp.tail())
  st.header('Ratios')
  ul.stWriteDf(ratioDf.round(3).tail())
  st.header('Weights')
  dwTail(dw)
  bt(script, dp, dw, yrStart)

def runCSSCore(yrStart, isAppend=False):
  und = 'FXI'
  dp, dw, dfDict, hv = btSetup([und],yrStart=yrStart-1)
  df = dfDict[und]
  if isAppend:
    yest=getYestNYSE()
    if (pendulum.instance(df.index[-1]).date() < yest.date()) and (pendulum.now().hour < MKT_CLOSE_HOUR + 1):
      ul.tPrint('Appending data to backtest ....')
      data = yf.Ticker(und).history(period='1d')
      df2 = df.tail(1).copy()
      df2.index = [yest]
      for field in ul.spl('Open,High,Low,Close,Volume'):
        df2.loc[yest, field] = float(data[field].iloc[0])
      df = pd.concat([df, df2])
      dp.loc[yest,und] = df2.loc[yest, 'Close']
      dw = dp.copy()
      dw[:] = np.nan
      dfDict[und] = df.copy()
    else:
      ul.tPrint('Continuing on with backtest ....')
  #####
  ibsS = getIbsS(df)
  ratio1S = df['Close'] / df['Close'].shift(5)
  ratio1S.rename('Ratio 1', inplace=True)
  ratio2S = df['Close'] / getKFMeans(df['Close'])
  ratio2S.rename('Ratio 2', inplace=True)
  isTomS = getTomS(dp[und], 0, 2, isNYSE=True)
  isEntryS = ((ibsS > .9) & (ratio1S > 1) & (ratio2S > 1) & (isTomS == 0)) * 1
  isExitS = (ibsS < (1 / 3))*1
  stateS = getStateS(isEntryS, isExitS, isCleaned=True, isMonthlyRebal=False)
  ul.cachePersist('w','tmp1',stateS)
  dw[und] = -stateS
  #dw.loc[~dw.index.month.isin([5, 6, 7, 8, 9, 10]), und] /= 2
  dw.loc[dw.index.year < yrStart] = 0
  d=dict()
  d['und'] = und
  d['dp'] = dp
  d['dw'] = dw
  d['dfDict'] = dfDict
  d['ibsS'] = ibsS
  d['ratio1S'] = ratio1S
  d['ratio2S'] = ratio2S
  d['isTomS'] = isTomS
  d['isEntryS'] = isEntryS
  d['isExitS'] = isExitS
  d['stateS'] = stateS
  return d

def runCSS(yrStart=START_YEAR_DICT['CSS'], isSkipTitle=False):
  script = 'CSS'
  if not isSkipTitle:
    st.header(script)
  d=runCSSCore(yrStart)
  st.header('Table')
  df=d['dfDict'][d['und']]
  tableS = ul.merge(df['Close'].round(2), df['High'].round(2), df['Low'].round(2),
                    d['ibsS'].round(3), d['ratio1S'].round(3), d['ratio2S'].round(3), d['isTomS'], d['stateS'].ffill(), how='inner')
  ul.stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runBTSCore(yrStart):
  und = 'BTC'
  volTgt = .24
  maxWgt = 1
  df = getPriceHistory(und, yrStart)
  dp = df[['Close']]
  dp.columns = [und]
  ratio1S = dp[und] / dp[und].shift(28)
  ratio1S.rename('Ratio 1', inplace=True)
  ratio2S = dp[und] / dp[und].rolling(5).mean()
  ratio2S.rename('Ratio 2', inplace=True)
  #####
  isTomS = getTomS(dp[und], -4, 3)
  #####
  momScoreS = (ratio1S >= 1) * 1 + (ratio2S >= 1) * 1
  stateS = ((momScoreS == 2) * 1 | (isTomS == 1) * 1) * 1
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
  d['isTomS']=isTomS
  d['stateS']=stateS
  return d

def runBTS(yrStart=START_YEAR_DICT['BTS'], isSkipTitle=False):
  script = 'BTS'
  if not isSkipTitle:
    st.header(script)
  d=runBTSCore(yrStart)
  st.header('Table')
  tableS = ul.merge(d['dp'][d['und']], d['ratio1S'].round(3), d['ratio2S'].round(3), d['isTomS'], d['stateS'], how='inner')
  ul.stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runARTCore(yrStart, isAppend=False):
  undE = 'SPY'
  undC = 'FXI'
  undB = 'TLT'
  undG = 'GLD'
  dp, dw, dfDict, hv = btSetup([undE, undC, undB, undG], yrStart=yrStart-1)
  #####
  oSE = dfDict[undE]['Open']
  hSE = dfDict[undE]['High']
  lSE = dfDict[undE]['Low']
  cSE = dfDict[undE]['Close']
  #####
  hSC = dfDict[undC]['High']
  lSC = dfDict[undC]['Low']
  cSC = dfDict[undC]['Close']
  #####
  cSB = dfDict[undB]['Close']
  #####
  hSG = dfDict[undG]['High']
  lSG = dfDict[undG]['Low']
  cSG = dfDict[undG]['Close']
  #####
  if isAppend:
    yest = getYestNYSE()
    if (pendulum.instance(dp.index[-1]).date() < yest.date()) and (pendulum.now().hour < MKT_CLOSE_HOUR + 1):
      ul.tPrint('Appending data to backtest ....')
      for und in [undE,undC,undB,undG]:
        data = yf.Ticker(und).history(period='1d')
        row = dfDict[und].tail(1).copy()
        row.index = [yest]
        for field in ul.spl('Open,High,Low,Close,Volume'):
          row.loc[yest, field] = float(data[field].iloc[0])
        dfDict[und] = pd.concat([dfDict[und], row])
        dp.loc[yest, und] = row.loc[yest, 'Close']
      dw = dp.copy()
      dw[:] = np.nan
    else:
      ul.tPrint('Continuing on with backtest ....')
  #####
  # SPY
  isBHSE = (pandas_ta.cdl_pattern(oSE, hSE, lSE, cSE, name='harami')['CDL_HARAMI'] == 100).rename('BH?') * 1
  isPriorDownSE = (cSE.shift(1)<cSE.shift(2)).rename('Prior Down')*1
  ratioSE = (cSE / cSE.rolling(200).mean()).rename('Ratio')
  rawSE = ((isBHSE == 1) & (isPriorDownSE==1) & (ratioSE >= 1)) * 1
  rawSE.rename('Raw',inplace=True)
  preStateSE1 = rawSE.rolling(5).sum().clip(None, 1).rename('Pre-State 1')
  #####
  df = pd.read_csv('https://www.sumgrowth.com/StormGuardData.csv')
  df.index = pd.to_datetime(df['Date'])
  sgArmorSE = applyDates(df['SG-Armor'], cSE).shift()
  wprSE = pandas_ta.willr(hSE, lSE, cSE, length=2).rename('WPR')
  isEntryS = ((wprSE < (-90)) & (sgArmorSE > 0))*1
  isExitS = (wprSE > (-10))*1
  preStateSE2 = getStateS(isEntryS, isExitS, isCleaned=False, isMonthlyRebal=False).rename('Pre-State 2')
  stateSE = (preStateSE1 + preStateSE2).clip(None,1).rename('State')
  #####
  # FXI
  ibsSC = getIbsS(dfDict[undC])
  ratio1SC = (cSC / cSC.shift(5)).rename('Ratio 1')
  ratio2SC = (cSC / getKFMeans(cSC)).rename('Ratio 2')
  isTomSC = getTomS(cSC, 0, 2, isNYSE=True)
  isEntrySC = ((ibsSC > .9) & (ratio1SC > 1) & (ratio2SC > 1) & (isTomSC == 0)) * 1
  isExitSC = (ibsSC < (1 / 3)) * 1
  stateSC = -getStateS(isEntrySC, isExitSC, isCleaned=False, isMonthlyRebal=False).rename('State')
  ul.cachePersist('w','tmp2',stateSC)
  #####
  # TLT
  w1S = getTomS(cSB, -7, 0 - 1, isNYSE=True)
  w2S = getTomS(cSB, 0, 7 - 1, isNYSE=True)
  stateSB = w1S-w2S
  stateSB.rename('State', inplace=True)
  #####
  # GLD
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
  preStateSG1 = getStateS(isEntryS, isExitS, isCleaned=False, isMonthlyRebal=False).rename('Pre-State 1')
  #####
  ibsSG = getIbsS(dfDict[undG])
  adxSG = pandas_ta.adx(hSG, lSG, cSG, length=5)['ADX_5'].rename('ADX5')
  cond4S = ((ibsSG < .15) & (adxSG > 30) & (cSG.index.day>=15)) * 1
  cond4S.rename('Condition 4',inplace=True)
  isEntryS = cond4S
  isExitS = (cSG > cSG.shift()) * 1
  preStateSG2 = getStateS(isEntryS, isExitS, isCleaned=False, isMonthlyRebal=False).rename('Pre-State 2')
  #####
  isSeasonalSG = getIsSeasonalS(cSG,12,27,1,26)
  #####
  stateSG=(preStateSG1+preStateSG2+isSeasonalSG).clip(None,1)
  stateSG.rename('State',inplace=True)
  #####
  dw[undE] = cleanS(stateSE, isMonthlyRebal=False) * 1
  dw[undC] = cleanS(stateSC, isMonthlyRebal=False) * 1
  dw[undB] = cleanS(stateSB, isMonthlyRebal=False) * 1
  dw[undG] = cleanS(stateSG, isMonthlyRebal=False) * 1
  dw.loc[dw.index.year < yrStart] = 0
  dwAllOrNone(dw)
  d=dict()
  d['undE']=undE
  d['undC']=undC
  d['undB']=undB
  d['undG']=undG
  d['dp'] = dp
  d['dw'] = dw
  d['dfDict'] = dfDict
  #####
  d['cSE'] = cSE
  d['isBHSE']=isBHSE
  d['isPriorDownSE']=isPriorDownSE
  d['ratioSE']=ratioSE
  d['rawSE']=rawSE
  d['preStateSE1']=preStateSE1
  d['wprSE']=wprSE
  d['sgArmorSE']=sgArmorSE
  d['preStateSE2']=preStateSE2
  d['stateSE'] = stateSE
  #####
  d['cSC'] = cSC
  d['hSC'] = hSC
  d['lSC'] = lSC
  d['ibsSC'] = ibsSC
  d['ratio1SC'] = ratio1SC
  d['ratio2SC'] = ratio2SC
  d['isTomSC'] = isTomSC
  d['isEntrySC'] = isEntrySC
  d['isExitSC'] = isExitSC
  d['stateSC'] = stateSC
  #####
  d['cSB'] = cSB
  d['stateSB'] = stateSB
  #####
  d['cSG'] = cSG
  d['hSG'] = hSG
  d['cond1S']=cond1S
  d['cond2S']=cond2S
  d['cond3S']=cond3S
  d['preStateSG1']=preStateSG1
  d['ibsSG']=ibsSG
  d['adxSG']=adxSG
  d['cond4S']=cond4S
  d['preStateSG2']=preStateSG2
  d['isSeasonalSG']=isSeasonalSG
  d['stateSG']=stateSG
  return d

def runART(yrStart=START_YEAR_DICT['ART'], isSkipTitle=False):
  script = 'ART'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runARTCore(yrStart)
  st.header('Tables')
  st.subheader(d['undE'])
  tableSE = ul.merge(d['cSE'].round(2),d['isBHSE'],d['isPriorDownSE'],d['ratioSE'].round(3),d['rawSE'],d['preStateSE1'],how='inner')
  ul.stWriteDf(tableSE.tail())
  tableSE2 = ul.merge(d['wprSE'].round(2),d['sgArmorSE'],d['preStateSE2'],d['stateSE'].ffill(), how='inner')
  ul.stWriteDf(tableSE2.tail())
  #####
  st.subheader(d['undC'])
  tableSC = ul.merge(d['cSC'].round(2), d['hSC'].round(2), d['lSC'].round(2),d['ibsSC'].round(3),d['ratio1SC'].round(3),d['ratio2SC'].round(3),d['isTomSC'],d['stateSC'].ffill(),how='inner')
  ul.stWriteDf(tableSC.tail())
  #####
  st.subheader(d['undB'])
  tableSB = ul.merge(d['cSB'].round(2),d['stateSB'].ffill(),how='inner')
  ul.stWriteDf(tableSB.tail())
  #####
  st.subheader(d['undG'])
  tableSG = ul.merge(d['cSG'].round(2),d['hSG'].round(2),d['cond1S'],d['cond2S'],d['cond3S'],d['preStateSG1'], how='inner')
  ul.stWriteDf(tableSG.tail())
  tableSG2 = ul.merge(d['ibsSG'].round(3),d['adxSG'].round(1),d['cond4S'],d['preStateSG2'],d['isSeasonalSG'],d['stateSG'].ffill(), how='inner')
  ul.stWriteDf(tableSG2.tail())
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
  ul.stWriteDf(df)
  #####
  # Calcs
  dp = pd.DataFrame()
  for strategy in strategies:
    dp[strategy] = ul.cachePersist('r', strategy)
  dp = applyDates(dp, dp.iloc[:,-1]).ffill()
  dw = dp * np.nan
  pe = endpoints(dw, 'ME')
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
  ul.stWriteDf(dp2, isMaxHeight=True)

def runCore(yrStart=START_YEAR_DICT['Core']):
  strategies = ul.spl('IBS,TPP')
  weights = [1 / 2, 1 / 2]
  script = 'Core'
  runAggregate(yrStart, strategies, weights, script)