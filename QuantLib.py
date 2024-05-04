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
GET_PRICE_HISTORY_START_YEAR=2011-5
YFINANCE_START_YEAR=2023
IBS_START_YEAR=2013
TPP_START_YEAR=2013
CORE_START_YEAR=2013
CSS_START_YEAR=2013
BTS_START_YEAR=2015
ART_START_YEAR=2013-5

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

def btSetup(tickers, hvN=32, yrStart=GET_PRICE_HISTORY_START_YEAR, applyDatesS=None):
  dfDict=dict()
  for und in tickers:
    df=getPriceHistory(und,yrStart=yrStart)
    cS=df['Close'].rename(und).to_frame()
    if not dfDict:
      dp=cS
    else:
      dp= ul.merge(dp, cS, how='outer')
    dfDict[und] = df
  dp=dp.ffill()
  dw=dp.copy()
  dw.values[:] = np.nan
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

#####
# Etc
#####
def applyDates(a,b):
  return a.reindex(b.index,method='pad').ffill().copy()

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
  return s[s.index <= dtLast]

def getYFinanceS(ticker):
  from_date = f"{YFINANCE_START_YEAR}-01-01"
  to_date = pendulum.today().format('YYYY-MM-DD')
  return yf.download(ticker, start=from_date, end=to_date)['Adj Close'].rename(ticker)

#############################################################################################

#########
# Scripts
#########
def runIBS(yrStart=IBS_START_YEAR):
  undE = 'SPY'
  undQ = 'QQQ'
  undB = 'TLT'
  volTgt = .16
  maxWgt = 1
  #####
  script = 'IBS'
  st.header(script)
  dp, dw, dfDict, hv = btSetup([undE, undQ, undB])
  #####
  def m(und, dfDict, isMondayS=None):
    df = dfDict[und]
    ibsS=getIbsS(df)
    if und==undE:
      isEntryS = (isMondayS == 1) & (ibsS < .2) & (df['Low'] < df['Low'].shift(1))
      isExitS = df['Close'] > df['High'].shift(1)
    elif und==undQ:
      isEntryS = ibsS < .1
      isExitS = df['Close'] > df['High'].shift(1)
    elif und==undB:
      isEntryS = (ibsS < .15) & (df['Low'] < df['Low'].shift(1))
      isExitS = ibsS > .55
    else:
      ul.iExit('runIBS')
    stateS = getStateS(isEntryS, isExitS, isCleaned=True, isMonthlyRebal=False)
    return ibsS, stateS
  #####
  isMondayS = dfDict[undE]['Close'].rename('Monday?') * 0
  isMondayS[isMondayS.index.weekday == 0] = 1
  ibsSE, stateSE = m(undE, dfDict, isMondayS=isMondayS)
  ibsSQ, stateSQ = m(undQ,dfDict)
  ibsSB, stateSB = m(undB,dfDict)
  dw[undE] = stateSE
  dw[undQ] = stateSQ
  dw[undB] = stateSB
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  dwAllOrNone(dw)
  st.header('Tables')
  #####
  def m(und, ibsS, df, stateS, isMondayS=None):
    st.subheader(und)
    df2=ul.merge(df['Close'].round(2), df['High'].round(2), df['Low'].round(2), ibsS.round(3), how='inner')
    if isMondayS is not None: df2=ul.merge(df2, isMondayS, how='inner')
    df2=ul.merge(df2, stateS.ffill(), how='inner')
    ul.stWriteDf(df2.tail())
  #####
  m(undE, ibsSE, dfDict[undE], stateSE, isMondayS=isMondayS)
  m(undQ, ibsSQ, dfDict[undQ], stateSQ)
  m(undB, ibsSB, dfDict[undB], stateSB)
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

def runCSS(yrStart=CSS_START_YEAR, isSkipTitle=False):
  und = 'FXI'
  script = 'CSS'
  if not isSkipTitle:
    st.header(script)
  #####
  dp, dw, dfDict, hv = btSetup([und])
  df = dfDict[und]
  ibsS = getIbsS(df)
  ratio1S = df['Close'] / df['Close'].shift(5)
  ratio1S.rename('Ratio 1', inplace=True)
  ratio2S = df['Close']/getKFMeans(df['Close'])
  ratio2S.rename('Ratio 2', inplace=True)
  isTomS = getTomS(dp[und], 0, 2, isNYSE=True).rename('TOM?')
  isEntryS = (ibsS > .9) & (ratio1S > 1) & (ratio2S>1) & (isTomS == 0)
  isExitS = ibsS < 1/3
  stateS = getStateS(isEntryS, isExitS, isCleaned=True, isMonthlyRebal=False)
  dw[und] = -stateS * .75
  dw.loc[dw.index.month.isin([5, 6, 7, 8, 9, 10]), und] *= 2
  dw.loc[dw.index.year < yrStart] = 0
  #####
  st.header('Table')
  tableS = ul.merge(df['Close'].round(2), df['High'].round(2), df['Low'].round(2), ibsS.round(3), ratio1S.round(3), ratio2S.round(3), isTomS, stateS.ffill(), how='inner')
  ul.stWriteDf(tableS.tail())
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
  ratio1S=dp[und]/dp[und].shift(28)
  ratio1S.rename('Ratio 1',inplace=True)
  ratio2S=dp[und]/dp[und].rolling(5).mean()
  ratio2S.rename('Ratio 2', inplace=True)
  #####
  isTomS = getTomS(dp[und], -4, 3).rename('TOM?')
  #####
  momScoreS=(ratio1S>=1)*1+(ratio2S>=1)*1
  stateS = ((momScoreS==2)*1 | (isTomS==1)*1)*1
  stateS.rename('State', inplace=True)
  #####
  dw=dp.copy()
  dw[und]=stateS
  dw=cleanS(dw, isMonthlyRebal=True)
  hv = getHV(dp,n=16,af=365)
  dw = (dw * volTgt**2 / hv**2).clip(0, maxWgt)
  #####
  st.header('Table')
  tableS = ul.merge(df['Close'], ratio1S.round(3), ratio2S.round(3), isTomS, stateS, how='inner')
  ul.stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(dw)
  bt(script, dp, dw, yrStart)

def runART(yrStart=ART_START_YEAR, isSkipTitle=False):
  undE = 'SPY'
  undB = 'TLT'
  undG = 'GLD'
  #####
  script = 'ART'
  if not isSkipTitle:
    st.header(script)
  #####
  dp, dw, dfDict, hv = btSetup([undE, undB, undG])
  #####
  oSE = dfDict[undE]['Open']
  hSE = dfDict[undE]['High']
  lSE = dfDict[undE]['Low']
  cSE = dfDict[undE]['Close']
  #####
  cSB = dfDict[undB]['Close']
  #####
  hSG = dfDict[undG]['High']
  lSG = dfDict[undG]['Low']
  cSG = dfDict[undG]['Close']
  #####
  # SPY
  isBHS = (pandas_ta.cdl_pattern(oSE, hSE, lSE, cSE, name='harami')['CDL_HARAMI'] == 100).rename('BH?') * 1
  ratioS = (cSE / cSE.rolling(200).mean()).rename('Ratio')
  rawS = ((isBHS == 1) & (ratioS >= 1)) * 1
  rawS.rename('Raw',inplace=True)
  preStateSE1 = rawS.rolling(5).sum().clip(None, 1).rename('Pre-State 1')
  #####
  df = pd.read_csv('https://www.sumgrowth.com/StormGuardData.csv')
  df.index = pd.to_datetime(df['Date'])
  sgArmorS = applyDates(df['SG-Armor'], cSE).shift()
  wprS = pandas_ta.willr(hSE, lSE, cSE, length=2).rename('WPR')
  isEntryS = ((wprS < (-90)) & (sgArmorS > 0))*1
  isExitS = (wprS > (-10))*1
  preStateSE2 = getStateS(isEntryS, isExitS, isCleaned=False, isMonthlyRebal=False).rename('Pre-State 2')
  stateSE = (preStateSE1 + preStateSE2).clip(None,1).rename('State')
  #####
  # TLT
  w1S = getTomS(cSB, -7, 0 - 1, isNYSE=True)
  w2S = getTomS(cSB, 0, 7 - 1, isNYSE=True)
  stateSB = w1S-w2S
  stateSB.rename('State', inplace=True)
  #####
  # GLD
  ibsSG = getIbsS(dfDict[undG])
  adxSG = pandas_ta.adx(hSG, lSG, cSG, length=5)['ADX_5'].rename('ADX5')
  #####
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
  cond4S = ((ibsSG < .15) & (adxSG > 30) & (cSG.index.day>=15)) * 1
  cond4S.rename('Condition 4',inplace=True)
  isEntryS = cond4S
  isExitS = (cSG > cSG.shift()) * 1
  preStateSG2 = getStateS(isEntryS, isExitS, isCleaned=False, isMonthlyRebal=False).rename('Pre-State 2')
  #####
  stateSG=(preStateSG1+preStateSG2).clip(None,1)
  stateSG.rename('State',inplace=True)
  #####
  dw[undE] = cleanS(stateSE, isMonthlyRebal=False) * 1
  dw[undB] = cleanS(stateSB, isMonthlyRebal=False) * 1
  dw[undG] = cleanS(stateSG, isMonthlyRebal=False) * 1
  dwAllOrNone(dw)
  #####
  st.header('Tables')
  st.subheader(undE)
  tableSE = ul.merge(cSE.round(2),isBHS,ratioS.round(3),rawS,preStateSE1,how='inner')
  ul.stWriteDf(tableSE.tail())
  tableSE2 = ul.merge(wprS.round(2),sgArmorS,preStateSE2,stateSE.ffill(), how='inner')
  ul.stWriteDf(tableSE2.tail())
  #####
  st.subheader(undB)
  tableSB = ul.merge(cSB.round(2),stateSB.ffill(),how='inner')
  ul.stWriteDf(tableSB.tail())
  #####
  st.subheader(undG)
  tableSG = ul.merge(cSG.round(2),hSG.round(2),cond1S,cond2S,cond3S,preStateSG1, how='inner')
  ul.stWriteDf(tableSG.tail())
  tableSG2 = ul.merge(ibsSG.round(3),adxSG.round(1),cond4S,preStateSG2,stateSG.ffill(), how='inner')
  ul.stWriteDf(tableSG2.tail())
  #####
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

def runCore(yrStart=CORE_START_YEAR):
  strategies = ul.spl('IBS,TPP')
  weights = [1 / 2, 1 / 2]
  script = 'Core'
  runAggregate(yrStart, strategies, weights, script)