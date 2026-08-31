###############
# Quant Library
###############
import UtilLib as ul
from UtilLib import SHARED_DICT
import PriceLib as pl
import DBLib as dl
import streamlit as st
import numpy as np
import pandas as pd
import math
import pendulum
import pandas_ta_classic as ta
import pandas_market_calendars

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
      self.df=pl.getPriceHistory(self.und,yrStart=self.yrStart)
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

def getCrsiS(closeS, rsiPeriods=3, streakPeriods=2, rankPeriods=100):
  def wilderRsi(s, n):
    d = s.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    au = up.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    return 100 - (100 / (1 + au / ad))
  #####
  signs = np.sign(closeS.diff().fillna(0).to_numpy())
  streak = np.zeros(len(signs))
  for i in range(1, len(signs)):
    if signs[i] > 0:
      streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
    elif signs[i] < 0:
      streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
  streakS = pd.Series(streak, index=closeS.index)
  #####
  rocS = closeS.pct_change() * 100
  def percentRank(x):
    return 100.0 * np.sum(x[:-1] < x[-1]) / (len(x) - 1)
  rankS = rocS.rolling(rankPeriods + 1).apply(percentRank, raw=True)
  #####
  return ((wilderRsi(closeS, rsiPeriods) + wilderRsi(streakS, streakPeriods) + rankS) / 3).rename('CRSI')

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

def getNYSEEomS(idx):
  """Sessions-to-month-end on the real NYSE calendar (1 = last session).

  Current month is ranked through getNYSEMonthEnd, not through last printed
  price, so mid-month bars are not treated as EOM.
  """
  idx = pd.DatetimeIndex(idx)
  if idx.tz is not None:
    idx = idx.tz_localize(None)
  idx = idx.normalize()
  today = pd.Timestamp(pendulum.now('America/New_York').to_date_string())
  curEom = pd.Timestamp(getNYSEMonthEnd(offset=0)).normalize()
  cal = pandas_market_calendars.get_calendar('NYSE')
  spanStart = (pd.Timestamp(idx[0]) - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
  spanEnd = max(idx[-1], curEom, today).strftime('%Y-%m-%d')
  sessions = pd.DatetimeIndex([
    pd.Timestamp(x).date() for x in cal.schedule(start_date=spanStart, end_date=spanEnd).index
  ])
  s = pd.DataFrame({'ym': sessions.to_period('M')}, index=sessions)
  curYm = curEom.to_period('M')
  s = s.loc[(s['ym'] < curYm) | ((s['ym'] == curYm) & (s.index <= curEom))]
  eom = s.groupby('ym').cumcount(ascending=False) + 1
  return eom.reindex(idx).rename('EOM')

def getNYSEHolidayDates(start, end, prefixes):
  cal = pandas_market_calendars.get_calendar('NYSE')
  start, end = pd.Timestamp(start), pd.Timestamp(end)
  out = []
  for rule in cal.regular_holidays.rules:
    name = rule.name.lower()
    if any(name.startswith(p) for p in prefixes):
      out.extend(rule.dates(start, end))
  if len(out) == 0:
    return pd.DatetimeIndex([])
  return pd.DatetimeIndex(pd.to_datetime(out)).normalize().unique().sort_values()

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
  return pd.Timestamp(pd.Timestamp(sessions[targetIdx]).date())

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

def getStateS_minhold(isEntryS, isExitS, minDays, isCleaned=False, isMonthlyRebal=True):
  if len(isEntryS)!=len(isExitS):
    ul.iExit('getStateS_minhold')
  stateS=(isEntryS * np.nan).rename('State')
  state=0
  daysHeld=0
  for i in range(len(stateS)):
    if state==0 and isEntryS.iloc[i]:
      state=1; daysHeld=0
    elif state==1:
      daysHeld+=1
      if daysHeld>=minDays and isExitS.iloc[i]: state=0
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

def _getGSBSignalS(df):
  hS = df['High']
  lS = df['Low']
  cS = df['Close']
  sma50S = cS.rolling(50).mean()
  atrS = ta.atr(hS, lS, cS, length=186)
  bandS = atrS * 3.1
  countS = 0
  for k in range(1, 240):
    countS += ((cS > lS.shift(k) - bandS) & (cS < hS.shift(k) + bandS)) * 1
  srSmoothS = (countS / 239 * 100).rolling(81).mean()
  qqeS = EMA(ta.rsi(cS, length=14), 5)
  domD1S = pd.Series(df.index.day, index=df.index, dtype=float).shift(1)
  srD1S = srSmoothS.shift(1)
  qqeD1S = qqeS.shift(1)
  dS = qqeD1S.diff()
  risingPrevS = (sma50S.diff()>0).shift(1)
  lowerNowS = (domD1S.round(5) < srD1S.round(5)).rolling(5).sum()==5
  fallQqePrevS = (((dS<=0).rolling(3).sum()==3) & ((dS<0).rolling(3).sum()>=1)).shift(1)
  return (risingPrevS & lowerNowS & fallQqePrevS).rename('GSB') * 1.0

def runTPP2Core(yrStart):
  volTgt = .135
  maxWgt = 1.5
  etc=ul.spl('HYG,SPHB,SPLV,TIP,NDX.INDX')
  dp, dw, dfDict, hv = btSetup(ul.spl('SPY,IWM,GLD,UUP')+etc,yrStart=yrStart-1)
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
    hv = hv.drop(und2, axis=1)
  ##############
  # SPY canaries
  ##############
  # 1. BTC
  btcS = pl.getPriceHistoryCrypto('BTC', yrStart - 1)['Close'].rename('BTC')
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

  # 5. VIX
  vixS = pl.getPriceHistory('VIX.INDX', yrStart=yrStart - 1)['Close']
  isCanaryS_VIX_Raw = (vixS < vixS.rolling(10).mean()).rename('VIX Raw') * 1.0
  isCanaryS_VIX = applyDates(isCanaryS_VIX_Raw.rolling(3).sum() >= 2, dp).rename('VIX') * 1.0

  # 6. GSB
  isCanaryS_GSB = applyDates(_getGSBSignalS(dfDict['SPY']), dp) * 1.0

  # Voting
  voteDf = pd.DataFrame({
    'BTC': isCanaryS_BTC,
    'HYG': isCanaryS_HYG,
    'SPHB_LV': isCanaryS_SPHB_LV,
    'TIP': isCanaryS_TIP,
    'VIX': isCanaryS_VIX,
    'GSB': isCanaryS_GSB,
  })
  voteCountS = voteDf.sum(axis=1).rename('Votes')
  dw['SPY'] = (voteCountS >= 3) * (voteCountS / 3)

  #####
  # IWM
  #####
  ndxS = applyDates(dfDict['NDX.INDX']['Close'],dp)
  isNDXOkS_IWM = ((ndxS < ndxS.rolling(5).max()).rolling(5).sum() == 5).rename('NDX Ok?') * 1
  ibsS_IWM = getIbsS(dfDict['IWM'])
  isIBSOkS_IWM = (ibsS_IWM == ibsS_IWM.rolling(8).min()).rename('IBS Ok?') * 1
  cS_IWM = dfDict['IWM']['Close']
  ratio200S_IWM = (cS_IWM / cS_IWM.rolling(200).mean()).rename('IWM Ratio 200D')
  isEntryS_IWM = applyDates((isNDXOkS_IWM==1) & (isIBSOkS_IWM==1) & (ratio200S_IWM>1), dp)
  dw['IWM'] = getStateS_timestop(isEntryS_IWM, isEntryS_IWM*0, 8, isCleaned=True, isMonthlyRebal=True)

  #####
  # GLD
  #####
  gldS = applyDates(dfDict['GLD']['Close'], dp)
  ratio150S_GLD = (gldS / gldS.rolling(150).mean()).rename('GLD Ratio 150D')
  df = dfDict['GLD']
  hS = df['High']
  lS = df['Low']
  cS = df['Close']
  ibsS = getIbsS(df)
  adxS = ta.adx(hS, lS, cS, length=5)['ADX_5']
  isEntryS = (cS > hS.rolling(3).max().shift()) | ((ibsS < .15) & (adxS > 30))
  isExitS = (cS > hS.shift()) | ((cS > cS.shift()) & (cS.shift() > cS.shift(2)))
  gtsSignalS = getStateS_minhold(isEntryS, isExitS, 1, isCleaned=False, isMonthlyRebal=False).rename('GTS Signal')
  m = lambda n: applyDates(n, dw) * 1
  dw['GLD'] = m(ratio150S_GLD > 1) / 2 + m(gtsSignalS) / 2

  #####
  # UUP
  #####
  uupS = applyDates(dfDict['UUP']['Close'], dp)
  ratio50S_UUP = (uupS / uupS.rolling(50).mean()).rename('UUP Ratio 50D')
  dw['UUP'] = m(ratio50S_UUP > 1)

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
  d['isCanaryS_VIX_Raw'] = isCanaryS_VIX_Raw
  d['isCanaryS_VIX'] = isCanaryS_VIX
  d['isCanaryS_GSB'] = isCanaryS_GSB
  #####
  d['isNDXOkS_IWM'] = isNDXOkS_IWM
  d['isIBSOkS_IWM'] = isIBSOkS_IWM
  d['ratio200S_IWM'] = ratio200S_IWM
  #####
  d['ratio150S_GLD']=ratio150S_GLD
  d['gtsSignal'] = gtsSignalS
  #####
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
    d['isCanaryS_SPHB_LV'], d['isCanaryS_TIP'], d['isCanaryS_VIX_Raw'], d['isCanaryS_VIX'], d['isCanaryS_GSB'],
    d['voteCountS'], how='inner').tail())
  st.header('IWM Filters')
  stWriteDf(ul.merge(d['isNDXOkS_IWM'],d['isIBSOkS_IWM'],d['ratio200S_IWM'].round(3),how='inner').tail())
  st.header('GLD/UUP Table')
  stWriteDf(ul.merge(d['ratio150S_GLD'].round(3), d['gtsSignal'], d['ratio50S_UUP'].round(3), how='inner').tail())
  st.header('States')
  stWriteDf(d['stateDf'].tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

#####

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
  vixS = applyDates(pl.getPriceHistory('VIX.INDX',yrStart=yrStart-1)['Close'].rename('VIX'),cS)
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

#####

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
  dw*=0.5 # weight reduction
  st.header('Prices')
  stWriteDf(dp.tail())
  st.header('Weights')
  dwTail(dw)
  bt(script, dp, dw, yrStart)

def runVCACore(yrStart):
  und='VIXM'
  etc=ul.spl('SPY,VIX.INDX')
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
  vixRatioS = (vixS / vixS.rolling(10).mean()).rename('VIX Ratio')
  hvS = (spyS.pct_change().rolling(10).std() * math.sqrt(252) * 100).rename('HV')
  eVRPS= (vixS-hvS).rename('eVRPS')
  eVRPS_pctl = eVRPS.rolling(252).rank(pct=True).rename('eVRPS Pctl')
  #####
  m= lambda s: applyDates(s,dw).ffill().fillna(0)
  w1 = m((spyRatioS < 1) & (ibsS > 0.75) & (vixRatioS > 1))
  w2 = m((eVRPS_pctl <= 0.25) & (vixRatioS > 1))
  dw[und] = cleanS((w1 + w2).clip(upper=1), isMonthlyRebal=False)
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
  return d

def runVCA(yrStart,isSkipTitle=False):
  script = 'VCA'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runVCACore(yrStart)
  st.header('Tables')
  tableS = ul.merge(d['dp'], d['SPY'], d['spyRatioS'].round(3), d['ibsS'].round(3),
                    d['VIX'], d['vixRatioS'].round(3), d['hvS'].round(2), d['eVRPS'].round(2), (d['eVRPS_pctl'] * 100).round(1), how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

#####

def runBTSCore(yrStart):
  volTgt = .255
  maxWgt = 1.5
  cS = pl.getPriceHistoryCrypto('BTC', yrStart=yrStart)['Close']
  ratioS = (cS / cS.rolling(50).mean()).rename('Ratio')
  ratio2S = (cS / cS.rolling(240).max()).rename('Ratio2')
  dw = ((ratioS > 1) & (ratio2S > 0.8)).rename('BTC').to_frame()
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

#####

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

#####

def _comUSOSignal(usoDf, ref, yrStart):
  oS, hS, lS, cS = usoDf['Open'], usoDf['High'], usoDf['Low'], usoDf['Close']
  dxyS = pl.getPriceHistory('DXY.INDX', yrStart=yrStart - 1)['Close']
  atrPctS = ta.atr(hS, lS, cS, length=1) / cS * 100
  crsiS = getCrsiS(cS)
  isCondS = oS < oS.shift(1)
  isCond2S = dxyS.rolling(120).mean() / dxyS.rolling(200).mean() < 1
  isCond3S = atrPctS > (atrPctS.rolling(100).mean() + atrPctS.rolling(100).std())
  isEntryS = applyDates(isCondS & isCond2S & isCond3S, ref)
  isExitS = applyDates(crsiS > 65, ref)
  return getStateS_minhold(isEntryS, isExitS, 1, isCleaned=False, isMonthlyRebal=False).rename('USO Signal')

def _comPLSignal(plDf, ref):
  cS, hS, lS = plDf['Close'], plDf['High'], plDf['Low']
  atr4S = ta.atr(hS, lS, cS, length=4) / cS * 100
  isEntryS = applyDates(hS >= (cS.shift(1) * (1 + 1.7 * atr4S.shift(1) / 100)), ref)
  return getStateS_timestop(isEntryS, isEntryS * 0, 3, isCleaned=False, isMonthlyRebal=False).rename('PL Signal')

def runCOMCore(yrStart):
  volTgt = .32
  wDict = {'USO':1/2,'PL':1/2}
  dp, _, dfDict, _ = btSetup(['USO'], yrStart=yrStart - 1)
  plDf = dl.getPriceHistoryDB('PL', yrStart=yrStart - 1)
  dp['PL'] = applyDates(plDf['Close'], dp)
  hv = getHV(dp)
  dw = dp.copy()
  #####
  dw['USO'] = _comUSOSignal(dfDict['USO'], dp, yrStart)
  dw['PL'] = _comPLSignal(plDf, dp)
  dw = cleanS(dw, isMonthlyRebal=True)
  dw = dw * (volTgt / hv).clip(0, 1)
  for und in wDict.keys():
    dw[und] *= wDict[und]
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  return d

def runCOM(yrStart, isSkipTitle=False):
  script = 'COM'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runCOMCore(yrStart)
  st.header('Prices')
  stWriteDf(d['dp'].tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

#####

def runCOSCore(yrStart):
  und, size = 'SPY', 2
  df = pl.getPriceHistory(und, yrStart=yrStart - 1)
  o, c = df['Open'], df['Close']
  ibsS = getIbsS(df)
  ddS = ((c < o) & (o < c.shift(2)) & (ibsS < .3) & (df.index.dayofweek <= 1)).fillna(False)
  seasonalS = ((df.index.day == 25) & (c < c.shift(1))).fillna(False)
  stateS = (ddS | seasonalS).rename('State')
  #####
  openIdx = df.index + pd.Timedelta(hours=9, minutes=30)
  closeIdx = df.index + pd.Timedelta(hours=16)
  px = pd.concat([
    pd.Series(o.values, index=openIdx, name=und),
    pd.Series(c.values, index=closeIdx, name=und),
  ]).sort_index().to_frame()
  dwI = pd.DataFrame(0.0, index=px.index, columns=[und])
  dwI.loc[closeIdx[stateS.values], und] = size
  dwI.loc[dwI.index.year < yrStart] = 0
  #####
  r = (stateS.astype(float).shift(1) * size * (o / c.shift(1) - 1)).fillna(0)
  r.loc[r.index.year < yrStart] = 0
  dp = (1 + r).cumprod().rename('COS').to_frame()
  dw = dp * np.nan
  dw.iloc[endpoints(dw), 0] = 1
  #####
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['dpIntraday'] = px
  d['dwIntraday'] = dwI
  d['df'] = df
  d['stateS'] = stateS.astype(float)
  return d

def runCOS(yrStart, isSkipTitle=False):
  script = 'COS'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runCOSCore(yrStart)
  st.header('Table')
  tableS = ul.merge(
    d['df']['Open'].round(2), d['df']['Close'].round(2), d['stateS'], how='inner')
  stWriteDf(tableS.tail())
  st.header('Weights')
  dwTail(d['dwIntraday'])
  bt(script, d['dp'], d['dw'], yrStart)

def runGEOCore(yrStart):
  volTgt = .32
  #####
  wDict = {'NATO.LSE': .5, 'NUCL.LSE': .5}
  unds = list(wDict.keys())
  etc = ul.spl('ITA,U-UN.TO')
  dp, dw, dfDict, hv = btSetup(unds + etc, yrStart=yrStart - 1)
  dp2 = dp.copy()
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
    hv = hv.drop(und2, axis=1)
  #####
  # NATO.LSE signal
  itaS = dfDict['ITA']['Close']
  itaMS = itaS.iloc[endpoints(itaS)]
  ratio12S_ITA = (itaMS / itaMS.rolling(12).mean()).rename('ITA Ratio 12M')
  dw['NATO.LSE'] = applyDates(ratio12S_ITA > 1, dw)
  #####
  # NUCL.LSE signal
  uunS = dfDict['U-UN.TO']['Close']
  uunMS = uunS.iloc[endpoints(uunS)]
  rocS_UUN = uunMS.pct_change().rename('UUN ROC 1M')
  dw['NUCL.LSE'] = applyDates(rocS_UUN > 0, dw)
  #####
  dw = cleanS(dw, isMonthlyRebal=True)
  dw = dw * (volTgt / hv).clip(0, 1)
  for und in unds:
    dw[und] *= wDict[und]
  #####
  d = dict()
  d['dp'] = dp
  d['dp2'] = dp2
  d['dw'] = dw
  d['ratio12S_ITA'] = ratio12S_ITA
  d['rocS_UUN'] = rocS_UUN
  return d

def runGEO(yrStart, isSkipTitle=False):
  script = 'GEO'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runGEOCore(yrStart)
  st.header('Table')
  stWriteDf(ul.merge(d['dp2'], d['ratio12S_ITA'].round(3), d['rocS_UUN'].round(3), how='inner').tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

def runSSSCore(yrStart):
  dp, dw, _, _ = btSetup(ul.spl('GLD,IEI'), yrStart=yrStart - 1)
  idx = dw.index
  dw[:]=0
  #####
  # IEI
  #####
  eomS = getNYSEEomS(idx)
  dw['IEI'] = ((eomS >= 2) & (eomS <= 4))*2
  #####
  # GLD
  #####
  for H in getNYSEHolidayDates(idx[0], idx[-1], ('christmas', 'new years')):
    before = idx[idx < H]
    if len(before) == 0:
      continue
    i0 = idx.get_loc(before[-1])
    dw.loc[idx[max(i0 - 2, 0): i0 + 1], 'GLD'] = 1
  ####
  # RB
  ####
  rbDf = dl.getPriceHistoryDB('RB', yrStart=yrStart - 1)
  dw['RB'] = 0.0
  rbx = rbDf.index
  for H in getNYSEHolidayDates(idx[0], idx[-1], ('memorial day', 'july 4', 'labor day', 'thanksgiving')):
    after = idx[idx > H]
    if not len(after):
      continue
    before = idx[idx < after[0]]
    if not len(before):
      continue
    d0, d1 = before[-1], after[0]
    ar, pad = rbx[rbx > H], rbx[rbx <= d0]
    if len(pad) and len(ar) and rbDf.loc[ar[0], 'Open'] > 0:
      rbDf.loc[pad[-1], 'Close'] = rbDf.loc[ar[0], 'Open']
    dw.loc[d0, 'RB'] = -0.5
    dw.loc[d1, 'RB'] = 0.0
  dp['RB'] = applyDates(rbDf['Close'], dp)
  #####
  dw = cleanS(dw, isMonthlyRebal=True)
  #####
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  return d

def runSSS(yrStart, isSkipTitle=False):
  script = 'SSS'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runSSSCore(yrStart)
  st.header('Prices')
  dwTail(d['dp'])
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

#####

def runHNXCore(yrStart):
  und = '000660.KO'
  und2 = '138230.KO'
  dp, dw, dfDict, hv = btSetup([und, und2], yrStart=yrStart - 1)
  dp2 = dp.copy()
  #####
  ibsS = getIbsS(dfDict[und])
  ibs2S = getIbsS(dfDict[und], n=2).rename('IBS2')
  cS = dfDict[und]['Close']
  ratioS = (cS / cS.rolling(200).mean()).rename('Ratio')
  rsiS = ta.rsi(cS, length=2).rename('RSI')
  isEntryS = (ibsS < .2) & (ibs2S < .4) & (ratioS > 1)
  isExitS = (ibsS > .5) | (rsiS > 70)
  dw[und] = getStateS_timestop(isEntryS, isExitS, 4, isCleaned=True, isMonthlyRebal=False)
  dw[und2] = -dw[und]
  dw.loc[dw.index.year < yrStart] = 0
  d = dict()
  d['dp'] = dp
  d['dp2'] = dp2
  d['dw'] = dw
  d['dfDict'] = dfDict
  d['ibsS'] = ibsS
  d['ibs2S'] = ibs2S
  d['ratioS'] = ratioS
  d['rsiS'] = rsiS
  return d

def runHNX(yrStart, isSkipTitle=False):
  script = 'HNX'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runHNXCore(yrStart)
  st.header('Table')
  und='000660.KO'
  df = d['dfDict'][und]
  dp2=d['dp2']
  zfmt='{:,.0f}'.format
  dp2[und]=dp2[und].round().map(zfmt)
  df2 = ul.merge(dp2,
                 df['High'].round().map(zfmt),
                 df['Low'].round().map(zfmt),
                 d['ibsS'].round(3), d['ibs2S'].round(3), d['ratioS'].round(3), d['rsiS'].round(1), how='inner')
  stWriteDf(df2.tail())
  st.header('Weights')
  dwTail(d['dw'])
  bt(script, d['dp'], d['dw'], yrStart)

#####


def _mmq_simulate(df, po, atr, window_end=1000, exit_opp=61.8, fill='next_open',
                  entry_thr=100.0, eod_et=1600, window_start=930):
  """Modified Milk: ±100 in / ±61.8 out, 9:30–10:00 ET, next-open fill, flat 16:00 ET.
  Skip new entries when 10m ATR14 on the signal bar is under 0.10% of close."""
  idx_et = df.index.tz_convert('America/New_York')
  tS = pd.Series(idx_et.hour * 100 + idx_et.minute, index=df.index)
  dates = pd.Series(idx_et.date, index=df.index)
  nxt_t, nxt_d = tS.shift(-1), dates.shift(-1)
  is_last_rth = ((tS < eod_et) & (
    nxt_d.isna() | (nxt_d != dates) | (nxt_t >= eod_et) | (nxt_t < tS - 200)
  )).to_numpy()
  in_win = ((tS >= window_start) & (tS < window_end)).to_numpy()
  prev = po.shift(1)
  long_sig = ((prev <= entry_thr) & (po > entry_thr)).to_numpy()
  short_sig = ((prev >= -entry_thr) & (po < -entry_thr)).to_numpy()
  long_x = ((prev >= -exit_opp) & (po < -exit_opp)).to_numpy()
  short_x = ((prev <= exit_opp) & (po > exit_opp)).to_numpy()
  po_ok = (~po.isna() & ~prev.isna()).to_numpy()
  quiet = (atr / df['Close'] * 100.0 < 0.10).fillna(False).to_numpy()
  t_et, dates_et = tS.to_numpy(), dates.to_numpy()
  opn, close = df['Open'].to_numpy(), df['Close'].to_numpy()
  n = len(df)
  pos = pending = 0
  entry_px, entry_i = np.nan, -1
  trades, daily_r = [], {}

  def close_trade(i, px, why):
    nonlocal pos, entry_px, entry_i
    pts = (px - entry_px) * pos
    r = pts / entry_px if entry_px else 0.0
    day = pd.Timestamp(dates_et[i])
    trades.append(dict(
      date=day, side=int(pos),
      entry=float(entry_px), exit=float(px), pts=float(pts), r=float(r),
      why=why, entry_t=int(t_et[entry_i]) if entry_i >= 0 else -1, exit_t=int(t_et[i]),
    ))
    daily_r[day] = (1.0 + daily_r.get(day, 0.0)) * (1.0 + r) - 1.0
    pos, entry_px, entry_i = 0, np.nan, -1

  def same_rth(i, j):
    return j < n and dates_et[j] == dates_et[i] and t_et[j] < eod_et

  for i in range(1, n):
    if pending != 0 and pos == 0:
      if t_et[i] < eod_et and dates_et[i] == dates_et[i - 1]:
        pos, entry_px, entry_i = pending, float(opn[i]), i
      pending = 0
    if pos != 0 and po_ok[i] and ((pos == 1 and long_x[i]) or (pos == -1 and short_x[i])):
      if fill == 'close' or not same_rth(i, i + 1):
        close_trade(i, close[i], 'opp')
      else:
        close_trade(i + 1, opn[i + 1], 'opp')
    if pos != 0 and is_last_rth[i]:
      close_trade(i, close[i], 'eod')
      pending = 0
      continue
    if not po_ok[i] or not in_win[i]:
      continue
    if long_sig[i]:
      side = 1
    elif short_sig[i]:
      side = -1
    else:
      continue
    if pos == side:
      continue
    if pos == -side:
      close_trade(i, close[i], 'rev')
    if pos != 0:
      continue
    if quiet[i]:
      continue
    if fill == 'close':
      pos, entry_px, entry_i = side, float(close[i]), i
    else:
      pending = side
  if pos != 0:
    close_trade(n - 1, close[-1], 'eod_end')

  days = pd.DatetimeIndex(pd.unique(pd.to_datetime(dates_et))).tz_localize(None)
  r = pd.Series(0.0, index=days, name='r')
  if daily_r:
    r.update(pd.Series({pd.Timestamp(k): v for k, v in daily_r.items()}))
  return r.clip(-0.25, 0.25), pd.DataFrame(trades)


def runMMQCore(yrStart, live=True):
  df = dl.getPriceHistoryDBIntraday('NQ', yrStart=yrStart, intervalMins=10, live=live)
  cS = df['Close']
  atr = ta.atr(df['High'], df['Low'], cS, length=14)
  raw = (cS - EMA(cS, 21)) / (3.0 * atr.replace(0.0, np.nan)) * 100.0
  r, trades = _mmq_simulate(df, EMA(raw, 3), atr)
  r = r[r.index.year >= int(yrStart)]
  dp = (1 + r).cumprod().rename('MMQ').to_frame()
  dw = dp * np.nan
  dw.iloc[endpoints(dw), 0] = 1
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['trades'] = trades
  return d


def runMMQ(yrStart, isSkipTitle=False):
  script = 'MMQ'
  if not isSkipTitle:
    st.header(script)
  d = runMMQCore(yrStart)
  st.header('Trades')
  tr = d['trades']
  if tr is not None and len(tr):
    stWriteDf(tr.tail())
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

