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

###########
# Constants
###########
quandl.ApiConfig.api_key = st.secrets['quandl_api_key']
CC_API_KEY = st.secrets['cc_api_key']
GET_PRICE_HISTORY_START_YEAR=2011
YFINANCE_START_YEAR=2023
IBS_START_YEAR=2013
TPP_START_YEAR=2013
CORE_START_YEAR=2013
CSS_START_YEAR=2013
BTS_START_YEAR=2015

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
  ecTs = dp2.iloc[:, 0].rename('Equity Curve') * 0
  ec = ecTs[0] = 1
  p = dp2.iloc[0]
  w = dw2.iloc[0]
  for i in range(1, len(dp2)):
    r = dp2.iloc[i] / p - 1
    ecTs[i] = ec * (1 + sum(w * r))
    if not dw2.iloc[i].isnull().any():
      w = dw2.iloc[i]
      p = dp2.iloc[i]
      ec = ecTs[i]
  printCalendar(ecTs)
  nYears = (ecTs.index[-1] - ecTs.index[0]).days / 365
  cagr = math.pow(ecTs[-1] / ecTs[0], 1 / nYears) - 1
  dd = ecTs / ecTs.cummax() - 1
  upi = cagr / np.sqrt(np.power(dd, 2).mean())
  maxDD = -min(dd)
  vol = ((np.log(ecTs / ecTs.shift(1)) ** 2).mean()) ** 0.5 * (252 ** 0.5)

  m=lambda label,z: f"{label}: <font color='red'>{z}</font>"
  sep='&nbsp;'*10
  st.markdown(sep.join([
    m('&nbsp;'*3+'UPI', f"{upi:.2f}"),
    m('Sharpe', f"{cagr / vol:.2f}"),
    m('Cagr', f"{cagr:.1%}"),
    m('MaxDD', f"{maxDD:.1%}")
  ]), unsafe_allow_html=True)
  ul.cachePersist('w',script,ecTs)

def btSetup(tickers,hvN=32,yrStart=GET_PRICE_HISTORY_START_YEAR,applyDatesTs=None):
  dfDict=dict()
  for und in tickers:
    df=getPriceHistory(und,yrStart=yrStart)
    cTs=df['Close'].rename(und).to_frame()
    if not dfDict:
      dp=cTs
    else:
      dp= ul.merge(dp, cTs, how='outer')
    dfDict[und] = df
  dp=dp.fillna(method='pad')
  dw=dp.copy()
  dw.values[:] = np.nan
  hv = getHV(dp, n=hvN)
  if applyDatesTs is None:
    return dp,dw,dfDict,hv
  else:
    return applyDates(dp,applyDatesTs),applyDates(dw,applyDatesTs),dfDict,applyDates(hv,applyDatesTs)

def dwAllOrNone(dw):
  selection = dw.isnull().sum(axis=1).isin(list(range(1,len(dw.columns))))
  dw[selection] = dw.fillna(method='pad')[selection]

def dwTail(dw,n=5):
  ul.stWriteDf(round(dw.dropna().tail(n), 3))

def printCalendar(ts):
  def rgroup(r, groups):
    def rprod(n):
      return (n + 1).prod() - 1
    return r.groupby(groups).apply(rprod)
  #####
  r = ts.pct_change()[1:]
  df = pd.DataFrame(rgroup(r, r.index.strftime('%Y-%m-01')))
  df.columns = ['Returns']
  df.index = df.index.map(pendulum.parse)
  df['Year'] = df.index.strftime('%Y')
  df['Month'] = df.index.strftime('%b')
  df = pd.pivot_table(data=df, index='Year', columns='Month', values='Returns', fill_value=0)
  df = df[ul.spl('Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec')]
  df['Year'] = rgroup(r, r.index.year).values
  df = df.applymap(lambda n:f"{n*100:.1f}")
  height = (len(df)+1) * 35 + 3
  df=df.style.applymap(lambda z:f"color: {'red' if float(z)<0 else '#228B22'}")
  st.dataframe(df,height=height)

#############################################################################################

#####
# Etc
#####
def applyDates(a,b):
  return a.reindex(b.index,method='pad').fillna(method='pad').copy()

def cleanTs(ts,isMonthlyRebal=True):
  ts=ts.astype('float64').fillna(method='pad')
  tmp=ts.shift(1)
  if isinstance(ts,pd.DataFrame):
    for i in range(1,len(ts)):
      if ts.iloc[i].equals(tmp.iloc[i]):
        ts.iloc[i]=np.nan
    if isMonthlyRebal:
      pe=endpoints(ts,'M')
      ts.iloc[pe]=ts.fillna(method='pad').iloc[pe]
  else:
    for i in range(1,len(ts)):
      if ts[i]==tmp[i]:
        ts[i]=np.nan
    if isMonthlyRebal:
      pe=endpoints(ts,'M')
      ts[pe]=ts.fillna(method='pad')[pe]
  return ts

def EMA(ts,n):
  return ts.ewm(span=n,min_periods=n,adjust=False).mean().rename('EMA')

# https://quantstrattrader.wordpress.com/author/ikfuntech/
def endpoints(df, on='M', offset=0):
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
  tltTs = getYFinanceS('TLT')
  iefTs = getYFinanceS('IEF')
  zbTs = getYFinanceS('ZB=F')
  znTs = getYFinanceS('ZN=F')
  tnTs = getYFinanceS('TN=F')
  zb_tlt_beta=getBeta(zbTs, tltTs)
  zn_ief_beta=getBeta(znTs, iefTs)
  tn_ief_beta=getBeta(tnTs, iefTs)
  return zb_tlt_beta,zn_ief_beta,tn_ief_beta

def getCoreWeightsDf():
  lastUpdateDict = ul.cachePersist('r','CR')['lastUpdateDict']
  fmt='DDMMMYY'
  dts = [pendulum.from_format(dt, fmt) for dt in lastUpdateDict.values()]
  lastUpdate = max(dts).format(fmt)

  l = list()
  d = ul.cachePersist('r', 'CR')['IBSDict']
  ep = 1e-9
  ibsDict = {'SPY': 0,
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

def getHV(ts, n=32, af=252):
  if isinstance(ts,pd.DataFrame):
    hv = ts.copy()
    for col in hv.columns:
      hv[col].values[:] = getHV(hv[col], n=n, af=af)
    return hv
  else:
    variances=(np.log(ts / ts.shift(1)))**2
    return (EMA(variances,n)**.5*(af**.5)).rename(ts.name)

def getKFMeans(ts):
  kf = pykalman.KalmanFilter(n_dim_obs=1, n_dim_state=1,
                    initial_state_mean=0,
                    initial_state_covariance=1,
                    transition_matrices=[1],
                    observation_matrices=[1],
                    observation_covariance=1,
                    transition_covariance=0.05)
  means, _ = kf.filter(ts)
  return pd.Series(means.flatten(), index=ts.index)

def getPriceHistory(und,yrStart=GET_PRICE_HISTORY_START_YEAR):
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
    df['date'] = [pendulum.from_timestamp(ts).naive() for ts in df['time']]
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
  df = df.sort_values(by=['date'])
  return df

def getStateTs(isEntryTs,isExitTs,isCleaned=False,isMonthlyRebal=True):
  if len(isEntryTs)!=len(isExitTs):
    ul.iExit('getStateTs')
  stateTs=(isEntryTs*np.nan).rename('State')
  state=0
  for i in range(len(stateTs)):
    if state==0 and isEntryTs[i]:
      state=1
    if state==1 and isExitTs[i]:
      state=0
    stateTs[i]=state
  if isCleaned:
    stateTs=cleanTs(stateTs,isMonthlyRebal=isMonthlyRebal)
  return stateTs.astype(float)

def getTomTs(ts, offsetBegin, offsetEnd, isNYSE=False): # 0,0 means hold one day starting from monthend
  ts=ts.copy()
  dtLast=ts.index[-1]
  dtLast2=pendulum.instance(dtLast)
  if isNYSE:
    dts=pandas_market_calendars.get_calendar('NYSE').schedule(start_date=dtLast2, end_date=dtLast2.add(days=30)).index
  else:
    dts = [dtLast2.add(days=i).date() for i in range(30)]
    dts = pd.DatetimeIndex(pd.to_datetime(dts))
  ts = ts.reindex(ts.index.union(dts))
  ts[:]=0
  for i in range(offsetBegin, offsetEnd+1):
    ts[endpoints(ts, offset=i)]=1
  return ts[ts.index<=dtLast]

def getYFinanceS(ticker):
  from_date = f"{YFINANCE_START_YEAR}-01-01"
  to_date = pendulum.today().format('YYYY-MM-DD')
  return yf.download(ticker, start=from_date, end=to_date)['Adj Close'].rename(ticker)

#############################################################################################

#########
# Scripts
#########
def runIBS(yrStart=IBS_START_YEAR):
  undE = 'QQQ'
  undB = 'TLT'
  volTgt = .16
  maxWgt = 2
  #####
  script = 'IBS'
  st.header(script)
  dp, dw, dfDict, hv = btSetup([undE, undB])
  #####
  def m(df):
    df = round(df, 10)
    ibsTs = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    ibsTs.rename('IBS', inplace=True)
    isEntryTs = ibsTs < .1
    isExitTs = df['Close'] > df['High'].shift(1)
    stateTs = getStateTs(isEntryTs, isExitTs, isCleaned=True)
    return ibsTs, stateTs
  #####
  ibsTsE, stateTsE = m(dfDict[undE])
  ibsTsB, stateTsB = m(dfDict[undB])
  dw[undE] = cleanTs(stateTsE)
  dw[undB] = cleanTs(stateTsB)
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  dwAllOrNone(dw)
  st.header('Tables')
  #####
  def m(und, ibsTs, df, stateTs):
    st.subheader(und)
    ul.stWriteDf(ul.merge(round(ibsTs, 3), df['Close'], df['High'], stateTs.fillna(method='pad'),how='inner').tail())
  #####
  m(undE, ibsTsE, dfDict[undE], stateTsE)
  m(undB, ibsTsB, dfDict[undB], stateTsB)
  st.header('Weights')
  dwTail(dw)
  bt(script, dp, dw, yrStart)

def runTPP(yrStart=TPP_START_YEAR):
  tickers = ul.spl('SPY,QQQ,IEF,GLD,UUP')
  lookback = 32
  volTgt = .16
  maxWgt = 3
  ######
  script = 'TPP'
  st.header(script)
  ######
  dp, dw, dfDict, hv = btSetup(tickers)
  ratioDf = dp / dp.rolling(200).mean()
  isOkDf = (ratioDf >= 1) * 1
  wDf = (1 / hv) * isOkDf
  rDf = np.log(dp / dp.shift(1))
  for i in endpoints(rDf, 'M'):
    origin = i - lookback + 1
    if origin >= 0:
      prTs = rDf.iloc[origin:(i + 1)].multiply(wDf.iloc[i], axis=1).sum(axis=1)
      pHv = ((prTs ** 2).mean()) ** .5 * (252 ** .5)
      dw.iloc[i] = wDf.iloc[i] * volTgt / pHv
  dw.clip(0, maxWgt, inplace=True)
  st.header('Prices')
  ul.stWriteDf(dp.tail())
  st.header('Ratios')
  ul.stWriteDf(round(ratioDf, 4).tail())
  st.header('Weights')
  dwTail(dw)
  bt(script, dp, dw, yrStart)

def runCSS(yrStart=CSS_START_YEAR, isSkipTitle=False):
  und = 'FXI'
  script = 'CSS'
  if not isSkipTitle:
    st.header(script)
  #####
  dp, dw, dfDict, hv = btSetup([und])
  df = round(dfDict[und], 10)
  ibsTs = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
  ibsTs.rename('IBS', inplace=True)
  ratio1Ts = df['Close'] / df['Close'].shift(5)
  ratio1Ts.rename('Ratio 1', inplace=True)
  ratio2Ts = df['Close']/getKFMeans(df['Close'])
  ratio2Ts.rename('Ratio 2', inplace=True)
  isTomTs = getTomTs(dp[und], 0, 2,isNYSE=True).rename('TOM?')
  isEntryTs = (ibsTs > .9) & (ratio1Ts > 1) & (ratio2Ts>1) & (isTomTs == 0)
  isExitTs = ibsTs < 1/3
  stateTs = getStateTs(isEntryTs, isExitTs,isCleaned=True,isMonthlyRebal=False)
  dw[und] = -stateTs * .75
  dw[und].loc[dw.index.month.isin([5, 6, 7, 8, 9, 10])] *= 2
  dw.loc[dw.index.year < yrStart] = 0
  #####
  st.header('Table')
  tableTs = ul.merge(df['Close'], round(ibsTs, 3), round(ratio1Ts, 3), round(ratio2Ts, 3), isTomTs, stateTs.fillna(method='pad'), how='inner')
  ul.stWriteDf(tableTs.tail())
  st.header('Weights')
  dwTail(dw)
  bt(script, dp, dw, yrStart)

def runBTS(yrStart=BTS_START_YEAR, isSkipTitle=False):
  volTgt = .24
  maxWgt = 1
  und='BTC'
  #####
  script = 'BTS'
  if not isSkipTitle:
    st.header(script)
  #####
  df=getPriceHistory(und,yrStart)
  dp=df[['Close']]
  dp.columns=[und]
  ratio1Ts=dp[und]/dp[und].shift(28)
  ratio1Ts.rename('Ratio 1',inplace=True)
  ratio2Ts=dp[und]/dp[und].rolling(5).mean()
  ratio2Ts.rename('Ratio 2', inplace=True)
  #####
  isTomTs = getTomTs(dp[und],-4,3).rename('TOM?')
  #####
  momScoreTs=(ratio1Ts>=1)*1+(ratio2Ts>=1)*1
  stateTs = ((momScoreTs==2)*1 | (isTomTs==1)*1)*1
  stateTs.rename('State', inplace=True)
  #####
  dw=dp.copy()
  dw[und]=stateTs
  dw=cleanTs(dw)
  hv = getHV(dp,n=16,af=365)
  dw = (dw * volTgt**2 / hv**2).clip(0, maxWgt)
  #####
  st.header('Table')
  tableTs = ul.merge(df['Close'], round(ratio1Ts, 3), round(ratio2Ts, 3), isTomTs, stateTs, how='inner')
  ul.stWriteDf(tableTs.tail())
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
  ul.stWriteDf(df)
  #####
  # Calcs
  dp = pd.DataFrame()
  for strategy in strategies:
    dp[strategy] = ul.cachePersist('r', strategy)
  dp = applyDates(dp, dp.iloc[:,-1]).fillna(method='pad')
  dw = dp * np.nan
  pe = endpoints(dw, 'M')
  for i in range(len(weights)):
    dw[strategies[i]].iloc[pe] = weights[i]
  #####
  # Backtest
  bt(script, dp, dw, yrStart)
  #####
  # Recent performance
  st.header('Recent Performance')
  dp2 = dp.copy()
  dp2[script] = ul.cachePersist('r', script)
  dp2 = dp2[[script] + strategies]
  dp2 = round((dp2 / dp2.iloc[-1]).tail(23) * 100, 2)
  ul.stWriteDf(dp2, isMaxHeight=True)

def runCore(yrStart=CORE_START_YEAR):
  strategies = ul.spl('IBS,TPP')
  weights = [1 / 2, 1 / 2]
  script = 'Core'
  runAggregate(yrStart, strategies, weights, script)