###############
# Strat Library
###############
# Strategy run*/run*Core bodies. Shared plumbing stays in QuantLib.
import UtilLib as ul
from UtilLib import SHARED_DICT
import PriceLib as pl
import streamlit as st
import numpy as np
import pandas as pd
import math
import pendulum
import pandas_ta_classic as ta
import pandas_market_calendars
from pathlib import Path

import QuantLib as ql

#########
# Scripts — AR live order (AR.py `l` / run*):
# TAA | US Broad MRs: COS,CUT,IBS,IBS2,IGQ,MTQ,RSS,STK,YPS
# | Other MRs: COS2,HMR,JMR | SCI,VCA,VCA2,VCA3,VCA4 | BCS,NGS,SIS,WTS | AHS,GEO,ZBT
# (GMR shelved 2026-11-21 → Lab/live-extras/GMRLib.py)
# (+ TPP not in AR, kept first)
#########


#####
# TPP (not in AR)
#####
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
  dp, dw, dfDict, hv = ql.btSetup([undQ,undB,undG,undD],yrStart=yrStart-1)
  ratioDf = dp / dp.rolling(200).mean()
  isOkDf = (ratioDf >= 1) * 1
  wDf = (1 / hv) * isOkDf
  rDf = np.log(dp / dp.shift(1))
  for i in ql.endpoints(rDf):
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
  ql.stWriteDf(dp.tail())
  st.header('Ratios')
  ql.stWriteDf(ratioDf.round(3).tail())
  st.header('Weights')
  ql.dwTail(dw)
  ql.bt(script, dp, dw, yrStart)

#####

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
  qqeS = ql.EMA(ta.rsi(cS, length=14), 5)
  domD1S = pd.Series(df.index.day, index=df.index, dtype=float).shift(1)
  srD1S = srSmoothS.shift(1)
  qqeD1S = qqeS.shift(1)
  dS = qqeD1S.diff()
  risingPrevS = (sma50S.diff()>0).shift(1)
  lowerNowS = (domD1S.round(5) < srD1S.round(5)).rolling(5).sum()==5
  fallQqePrevS = (((dS<=0).rolling(3).sum()==3) & ((dS<0).rolling(3).sum()>=1)).shift(1)
  return (risingPrevS & lowerNowS & fallQqePrevS).rename('GSB') * 1.0

def _getTOMSignalS(df):
  """QS EOM calendar canary on SPY: +1 while in trade.

  Entry: calendar day 29/30/31 and day return < 0.
  Exit: two successive positive closes, or close >= entry * 1.01.
  """
  cS = df['Close']
  rS = cS.pct_change()
  isEntryS = (rS < 0) & cS.index.day.isin([29, 30, 31])
  stateS = pd.Series(0.0, index=cS.index, name='TOM')
  state = 0
  entryPx = np.nan
  upStreak = 0
  for i in range(len(stateS)):
    if state == 0 and bool(isEntryS.iloc[i]):
      state = 1
      entryPx = float(cS.iloc[i])
      upStreak = 0
    stateS.iloc[i] = state
    if state == 1:
      r = rS.iloc[i]
      if pd.notna(r) and r > 0:
        upStreak += 1
      else:
        upStreak = 0
      if float(cS.iloc[i]) >= entryPx * 1.01 or upStreak >= 2:
        state = 0
        entryPx = np.nan
        upStreak = 0
  return stateS

def _getGSSSignalS(idx, prefixes=('christmas', 'new years'), n=3):
  """1 on the last n NYSE sessions before each matching holiday.

  Holidays are taken through 31 Jan of next year so a live run in December
  still sees Christmas / New Year. The window is the exchange calendar, not
  the truncated price index, so Dec 20 of a three-day window is marked on
  Dec 20, not rewritten after the holiday.
  """
  idx = pd.DatetimeIndex(idx)
  raw = idx
  if idx.tz is not None:
    idx = idx.tz_localize(None)
  norm = idx.normalize()
  today = pd.Timestamp(pendulum.now('America/New_York').to_date_string())
  anchor = max(pd.Timestamp(norm[-1]), today)
  horizon = pd.Timestamp(year=anchor.year + 1, month=1, day=31)
  holidays = ql.getNYSEHolidayDates(norm[0], horizon, prefixes)
  cal = pandas_market_calendars.get_calendar('NYSE')
  span0 = (pd.Timestamp(norm[0]) - pd.Timedelta(days=20)).strftime('%Y-%m-%d')
  span1 = (horizon + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
  sess = pd.DatetimeIndex([
    pd.Timestamp(x).date() for x in cal.schedule(start_date=span0, end_date=span1).index
  ])
  s = pd.Series(0.0, index=raw).rename('GSS Signal')
  for H in holidays:
    before = sess[sess < pd.Timestamp(H).normalize()]
    if len(before) == 0:
      continue
    window = before[-n:]
    s.iloc[pd.DatetimeIndex(norm).isin(window)] = 1
  return s


#####
# TAA
#####
def runTAACore(yrStart):
  # 2026-11-18: GLD booth /3→/4 + VT 0.25→0.285 + HI thr 0.96 (LOO recover)
  volTgt = .285
  maxWgt = 1.5
  etc=ul.spl('HYG,SPHB,SPLV,TIP,TRIN.INDX,NDX.INDX')
  dp, dw, dfDict, hv = ql.btSetup(ul.spl('SPY,IWM,IEI,GLD,UUP')+etc,yrStart=yrStart-1)
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
  isCanaryS_BTC = ql.applyDates(ratio50S_BTC > 1, dp).rename('BTC') * 1.0

  # 2. HYG
  hygS = ql.applyDates(dfDict['HYG']['Close'],dp)
  ratio100S_HYG = (hygS / ql.EMA(hygS, 100)).rename('HYG Ratio 100D')
  isCanaryS_HYG = (ratio100S_HYG > 1).rename('HYG') * 1.0

  # 3. SPHB/LV momentum
  def m(s):
    sum = 0
    for i in range(13):
      sum += s.shift(i * 21)
    return 13 * s / sum - 1
  #####
  sphbS = ql.applyDates(dfDict['SPHB']['Close'],dp)
  splvS = ql.applyDates(dfDict['SPLV']['Close'],dp)
  sphb_lv_ratio = sphbS / splvS
  isCanaryS_SPHB_LV = (m(sphb_lv_ratio) > 0).rename('SPHB_LV') * 1.0

  # 4. TIP
  cS_TIP = ql.applyDates(dfDict['TIP']['Close'],dp)
  momS_TIP = (cS_TIP.pct_change(21) + cS_TIP.pct_change(63) + cS_TIP.pct_change(126) + cS_TIP.pct_change(252)) / 4
  isCanaryS_TIP = (momS_TIP > 0).rename('TIP') * 1.0

  # 5. VIX
  vixS = pl.getPriceHistory('VIX.INDX', yrStart=yrStart - 1)['Close']
  isCanaryS_VIX_Raw = (vixS < vixS.rolling(10).mean()).rename('VIX Raw') * 1.0
  isCanaryS_VIX = ql.applyDates(isCanaryS_VIX_Raw.rolling(3).sum() >= 2, dp).rename('VIX') * 1.0

  # 6. GSB
  isCanaryS_GSB = ql.applyDates(_getGSBSignalS(dfDict['SPY']), dp) * 1.0

  # 7. TOM (QS EOM cal: +1 while in SPY trade after neg day on cal 29/30/31)
  isCanaryS_TOM = ql.applyDates(_getTOMSignalS(dfDict['SPY']), dp) * 1.0

  # 8. TRIN (neg voter: SMA20 < 1 → -1 in the booth)
  trinS = ql.applyDates(dfDict['TRIN.INDX']['Close'], dp)
  isCanaryS_TRIN = (trinS.rolling(20).mean() < 1).rename('TRIN') * -1.0

  # 9. SPY_HI (neg voter: SPY Close within 2% of 63d high → -1 in the booth; dig 2026-11-08)
  spyHi63 = dfDict['SPY']['High'].rolling(63).max()
  isCanaryS_HI = ql.applyDates(
    dfDict['SPY']['Close'] >= spyHi63 * 0.98, dp).rename('SPY_HI') * -1.0

  # Voting
  spyVoteDf = pd.DataFrame({
    'BTC': isCanaryS_BTC,
    'HYG': isCanaryS_HYG,
    'SPHB_LV': isCanaryS_SPHB_LV,
    'TIP': isCanaryS_TIP,
    'VIX': isCanaryS_VIX,
    'GSB': isCanaryS_GSB,
    'TOM': isCanaryS_TOM,
    'TRIN': isCanaryS_TRIN,
    'SPY_HI': isCanaryS_HI,
  })
  spyVoteCountS = spyVoteDf.sum(axis=1).rename('SPY Votes')
  dw['SPY'] = (spyVoteCountS >= 3) * (spyVoteCountS / 3)

  #####
  # IWM
  #####
  ndxS = ql.applyDates(dfDict['NDX.INDX']['Close'],dp)
  isNDXOkS_IWM = ((ndxS < ndxS.rolling(5).max()).rolling(5).sum() == 5).rename('NDX Ok?') * 1
  ibsS_IWM = ql.getIbsS(dfDict['IWM'])
  isIBSOkS_IWM = (ibsS_IWM == ibsS_IWM.rolling(8).min()).rename('IBS Ok?') * 1
  cS_IWM = dfDict['IWM']['Close']
  ratio200S_IWM = (cS_IWM / cS_IWM.rolling(200).mean()).rename('IWM Ratio 200D')
  isEntryS_IWM = ql.applyDates((isNDXOkS_IWM==1) & (isIBSOkS_IWM==1) & (ratio200S_IWM>1), dp)
  dw['IWM'] = ql.getStateS_timestop(isEntryS_IWM, isEntryS_IWM*0, 8, isCleaned=True, isMonthlyRebal=True)

  #####
  # IEI
  #####
  idx = dw.index
  eomS = ql.getNYSEEomS(idx)
  dw['IEI'] = ((eomS >= 2) & (eomS <= 4))

  ##############
  # GLD canaries
  ##############
  m = lambda n: ql.applyDates(n, dw) * 1.0
  df = dfDict['GLD']
  hS, lS, cS = df['High'], df['Low'], df['Close']

  # 1. r150 (Close > SMA150)
  gldS = ql.applyDates(cS, dp)
  ratio150S_GLD = (gldS / gldS.rolling(150).mean()).rename('GLD Ratio 150D')
  isCanaryS_GLD_r150 = m(ratio150S_GLD > 1).rename('r150')

  # 2. GTS (breakout / IBS∩ADX thrust, minhold)
  ibsS = ql.getIbsS(df)
  adxS = ta.adx(hS, lS, cS, length=5)['ADX_5']
  isEntryS = (cS > hS.rolling(3).max().shift()) | ((ibsS < .15) & (adxS > 30))
  isExitS = (cS > hS.shift()) | ((cS > cS.shift()) & (cS.shift() > cS.shift(2)))
  gtsSignalS = ql.getStateS_minhold(isEntryS, isExitS, 1, isCleaned=False, isMonthlyRebal=False).rename('GTS Signal')
  isCanaryS_GLD_GTS = m(gtsSignalS).rename('GTS')

  # 3. GSS (Christmas / New Year window)
  gssSignalS = _getGSSSignalS(df.index)
  isCanaryS_GLD_GSS = m(gssSignalS).rename('GSS')

  # 4. GLD Near HHV63? (neg voter: thr 0.96 → -1; dig 2026-11-18 LOO recover)
  isCanaryS_GLD_HI = ql.applyDatesBoolean(ql.getIsGLDNearHHV63S(df, thr=.96), dw) * -1.0

  # Voting
  gldVoteDf = pd.DataFrame({
    'r150': isCanaryS_GLD_r150,
    'GTS': isCanaryS_GLD_GTS,
    'GSS': isCanaryS_GLD_GSS,
    'GLD_HI': isCanaryS_GLD_HI,
  })
  gldVoteCountS = gldVoteDf.sum(axis=1).rename('GLD Votes')
  dw['GLD'] = (gldVoteCountS >= 1) * (gldVoteCountS / 4)

  #####
  # UUP
  #####
  uupS = ql.applyDates(dfDict['UUP']['Close'], dp)
  ratio50S_UUP = (uupS / uupS.rolling(50).mean()).rename('UUP Ratio 50D')
  dw['UUP'] = m(ratio50S_UUP > 1)

  #####
  stateDf=dw.astype(float).ffill()
  dw=ql.cleanS(dw,isMonthlyRebal=True)
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  #####
  d=dict()
  d['dp']=dp
  d['dw']=dw
  d['stateDf']=stateDf
  d['btcS'] = btcS
  #####
  # SPY canaries
  #####
  d['spyVoteDf'] = spyVoteDf
  d['spyVoteCountS'] = spyVoteCountS
  d['isCanaryS_BTC'] = isCanaryS_BTC
  d['isCanaryS_HYG'] = isCanaryS_HYG
  d['isCanaryS_SPHB_LV'] = isCanaryS_SPHB_LV
  d['isCanaryS_TIP'] = isCanaryS_TIP
  d['isCanaryS_VIX_Raw'] = isCanaryS_VIX_Raw
  d['isCanaryS_VIX'] = isCanaryS_VIX
  d['isCanaryS_GSB'] = isCanaryS_GSB
  d['isCanaryS_TOM'] = isCanaryS_TOM
  d['isCanaryS_TRIN'] = isCanaryS_TRIN
  d['isCanaryS_HI'] = isCanaryS_HI
  #####
  # IWM
  d['isNDXOkS_IWM'] = isNDXOkS_IWM
  d['isIBSOkS_IWM'] = isIBSOkS_IWM
  d['ratio200S_IWM'] = ratio200S_IWM
  #####
  # GLD canaries
  d['gldVoteDf'] = gldVoteDf
  d['gldVoteCountS'] = gldVoteCountS
  d['isCanaryS_GLD_r150'] = isCanaryS_GLD_r150
  d['isCanaryS_GLD_GTS'] = isCanaryS_GLD_GTS
  d['isCanaryS_GLD_GSS'] = isCanaryS_GLD_GSS
  d['isCanaryS_GLD_HI'] = isCanaryS_GLD_HI
  d['ratio150S_GLD'] = ratio150S_GLD
  d['gtsSignal'] = gtsSignalS
  d['gssSignal'] = gssSignalS
  #####
  # UUP
  d['ratio50S_UUP']=ratio50S_UUP
  return d

def runTAA(yrStart, isSkipTitle=False):
  script = 'TAA'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runTAACore(yrStart)
  dp=d['dp']
  st.header('Prices')
  ql.stWriteDf(ul.merge(dp,d['btcS'],how='inner').tail())
  st.header('SPY Canaries')
  ql.stWriteDf(ul.merge(
    d['spyVoteCountS'],
    d['isCanaryS_BTC'], d['isCanaryS_HYG'],
    d['isCanaryS_SPHB_LV'], d['isCanaryS_TIP'], d['isCanaryS_VIX_Raw'], d['isCanaryS_VIX'], d['isCanaryS_GSB'],
    d['isCanaryS_TOM'], d['isCanaryS_TRIN'], d['isCanaryS_HI'], how='inner').tail())
  st.header('IWM')
  ql.stWriteDf(ul.merge(d['isNDXOkS_IWM'],d['isIBSOkS_IWM'],d['ratio200S_IWM'].round(3),how='inner').tail())
  st.header('GLD Canaries')
  ql.stWriteDf(ul.merge(
    d['gldVoteCountS'],
    d['isCanaryS_GLD_r150'], d['isCanaryS_GLD_GTS'], d['isCanaryS_GLD_GSS'],
    d['isCanaryS_GLD_HI'], how='inner').tail())
  st.header('UUP')
  ql.stWriteDf(d['ratio50S_UUP'].round(3).to_frame().tail())
  st.header('States')
  ql.stWriteDf(d['stateDf'].tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

#####
# Sleeve Core/run defs: keep definition order = AR.py presentation (run*) order.
# Within each group, scripts are alphabetically sorted.
# Groups: Momentum (TAA) | US Broad MRs | Other MRs | Hedges | Alt1 | Etc.
# Parked WRS → Lab/live-extras/WRSLib.py (not in this file).
#####

#####
# US Broad MRs
#####
def runCOSCore(yrStart):
  und, size = 'SPY', 2
  df = pl.getPriceHistory(und, yrStart=yrStart - 1)
  o, c = df['Open'], df['Close']
  ibsS = ql.getIbsS(df)
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
  dw.iloc[ql.endpoints(dw), 0] = 1
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
  ql.stWriteDf(tableS.tail())
  st.header('Weights')
  ql.dwTail(d['dwIntraday'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runCUTCore(yrStart):
  # QQQ Cutler RSI3 cross<10 → exit C>priorC | TS5; VT 0.32 / maxWgt 1.5.
  # Seated @b 2026-11-18 (Lab CUT freeze; Harness PASS 2017+).
  und = 'QQQ'
  volTgt = .32
  maxWgt = 1.5
  dp, dw, dfDict, hv = ql.btSetup([und], yrStart=yrStart - 1)
  #####
  df = dfDict[und]
  cS = df['Close']
  cutS = ql.getCutlerRsiS(cS, n=3)
  isEntryS = (cutS < 10) & (cutS.shift(1) >= 10)
  isExitS = cS > cS.shift(1)
  stateS = ql.getStateS_timestop(isEntryS, isExitS, 5, isCleaned=True, isMonthlyRebal=True)
  dw[und] = stateS
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['dfDict'] = dfDict
  d['cutS'] = cutS
  return d

def runCUT(yrStart, isSkipTitle=False):
  script = 'CUT'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runCUTCore(yrStart)
  st.header('Table')
  df = d['dfDict']['QQQ']
  ql.stWriteDf(ul.merge(df['Close'].round(2), d['cutS'].round(1), how='inner').tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runIBSCore(yrStart):
  und = 'QQQ'
  volTgt = .225
  maxWgt = 1.5
  dp, dw, dfDict, hv = ql.btSetup([und],yrStart=yrStart-1)
  #####
  df = dfDict[und]
  ibsS = ql.getIbsS(df)
  isEntryS = ibsS < .1
  isExitS = (ibsS > .9) | (df['Close'] > df['High'].shift(1))
  stateS = ql.getStateS_timestop(isEntryS, isExitS, 7, isCleaned=True, isMonthlyRebal=True)
  dw[und] = stateS
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  ql.dwAllOrNone(dw)
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
  ql.stWriteDf(df2.tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runIBS2Core(yrStart):
  # IWM IBS twin: entry IBS<0.1; exit IBS>0.8 | hibreak; TS6;
  # VT 0.255 / maxWgt 1.5 — ql.bt MDD<15% (2026-11-04; was 0.258 at DD cap).
  und = 'IWM'
  volTgt = .255
  maxWgt = 1.5
  dp, dw, dfDict, hv = ql.btSetup([und],yrStart=yrStart-1)
  #####
  df = dfDict[und]
  ibsS = ql.getIbsS(df)
  isEntryS = ibsS < .1
  isExitS = (ibsS > .8) | (df['Close'] > df['High'].shift(1))
  stateS = ql.getStateS_timestop(isEntryS, isExitS, 6, isCleaned=True, isMonthlyRebal=True)
  dw[und] = stateS
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  ql.dwAllOrNone(dw)
  d=dict()
  d['dp']=dp
  d['dw']=dw
  d['dfDict']=dfDict
  d['ibsS']=ibsS
  return d

def runIBS2(yrStart,isSkipTitle=False):
  script = 'IBS2'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runIBS2Core(yrStart)
  st.header('Table')
  df = d['dfDict']['IWM']
  df2 = ul.merge(df['Close'].round(2), df['High'].round(2), df['Low'].round(2), d['ibsS'].round(3), how='inner')
  ql.stWriteDf(df2.tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runIGQCore(yrStart):
  # IGV signal → long QQQ. 5d<-7%; exit IGV C>priorH | TS5; VT 0.32 / maxWgt 1.5.
  # Seated @b 2026-11-19 (Lab IGQ freeze).
  und = 'QQQ'
  sig = 'IGV'
  volTgt = .32
  maxWgt = 1.5
  dp, dw, dfDict, hv = ql.btSetup([und, sig], yrStart=yrStart - 1)
  dp = dp.drop(sig, axis=1)
  dw = dw.drop(sig, axis=1)
  hv = hv.drop(sig, axis=1)
  #####
  df = dfDict[sig]
  cS = df['Close']
  r5S = cS.pct_change(5).rename('IGV 5d')
  isEntryS = r5S < -0.07
  isExitS = cS > df['High'].shift(1)
  stateS = ql.getStateS_timestop(isEntryS, isExitS, 5, isCleaned=True, isMonthlyRebal=True)
  dw[und] = stateS
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['dfDict'] = dfDict
  d['r5S'] = r5S
  return d

def runIGQ(yrStart, isSkipTitle=False):
  script = 'IGQ'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runIGQCore(yrStart)
  st.header('Table')
  df = d['dfDict']['IGV']
  ql.stWriteDf(ul.merge(
    df['Close'].round(2), df['High'].round(2),
    (100 * d['r5S']).round(2).rename('5d%'),
    how='inner',
  ).tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runMTQCore(yrStart):
  # MTUM signal → long QQQ. WPR7<-80 & IBS<.2; exit C>H[-1] | IBS>.9 | TS7; vt .32 / max1.5
  # Seated @b 2026-11-11 as MTM; renamed MTQ 2026-11-19. IBS entry .3→.2 2026-11-11.
  und = 'QQQ'
  etc = ['MTUM']
  volTgt = .32
  maxWgt = 1.5
  dp, dw, dfDict, hv = ql.btSetup([und] + etc, yrStart=yrStart - 1)
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
    hv = hv.drop(und2, axis=1)
  #####
  df = dfDict['MTUM']
  wprS = ta.willr(df['High'], df['Low'], df['Close'], length=7)
  ibsS = ql.getIbsS(df)
  isEntryS = (wprS < -80) & (ibsS < .2)
  isExitS = (df['Close'] > df['High'].shift(1)) | (ibsS > .9)
  stateS = ql.getStateS_timestop(isEntryS, isExitS, 7, isCleaned=True, isMonthlyRebal=True)
  dw[und] = stateS
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['dfDict'] = dfDict
  d['wprS'] = wprS
  d['ibsS'] = ibsS
  return d

def runMTQ(yrStart, isSkipTitle=False):
  script = 'MTQ'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runMTQCore(yrStart)
  st.header('Table')
  ql.stWriteDf(ul.merge(d['wprS'].rename('WPR7').round(1), d['ibsS'].rename('IBS').round(3), how='inner').tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runRSSCore(yrStart):
  und='SPY'
  dp, dw, dfDict, hv = ql.btSetup([und],yrStart=yrStart-1)
  #####
  cS = dfDict[und]['Close']
  vixS = ql.applyDates(pl.getPriceHistory('VIX.INDX',yrStart=yrStart-1)['Close'].rename('VIX'),cS)
  rsiS = ta.rsi(cS, length=2).rename('RSI2')
  ibsS = ql.getIbsS(dfDict[und],4)
  ratioS = (cS/cS.rolling(200).mean()).rename('Ratio')
  vixRatioS = (vixS.rolling(40).mean()/vixS.rolling(65).mean()).rename('VIX Ratio')
  #####
  isEntryS = (rsiS < 25) & (ibsS<.3) & (ratioS>1) & (vixRatioS<1)
  isExitS = (rsiS > 75) | (ibsS>.7) | (ratioS<=1) | (vixRatioS>=1)
  stateS = ql.getStateS(isEntryS, isExitS, isCleaned=True, isMonthlyRebal=True)
  #####
  # Summary
  dw[und] = stateS * 2
  #dw.loc[dw.index.year < yrStart] = 0
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
  ql.stWriteDf(tableS.tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runSTKCore(yrStart):
  # SPY Connors 2009 3-bar H/L streak: long 3↓H&↓L + Ratio250>1 + Ratio5<1;
  # short mirror; exit Ratio5 cross | TS9 fallback fuse (primary=SMA5);
  # same-close; 1x long / 2x short; no VT.
  # Dense daily dw (like RSS) — ql.cleanS monthly rebal mis-compounds L/S in ql.bt().
  # TS dig 2026-11-06: TS9 preferred vs TS8 (avoids 2011 day-8 crash cut); TS10 bounce-cut.
  und = 'SPY'
  dp, dw, dfDict, hv = ql.btSetup([und], yrStart=yrStart - 2)
  df = dfDict[und]
  hS, lS, cS = df['High'], df['Low'], df['Close']
  ratio5S = (cS / cS.rolling(5).mean()).rename('Ratio 5D')
  ratio250S = (cS / cS.rolling(250).mean()).rename('Ratio 250D')
  isDown3S = (
    (hS < hS.shift(1)) & (hS.shift(1) < hS.shift(2)) & (hS.shift(2) < hS.shift(3))
    & (lS < lS.shift(1)) & (lS.shift(1) < lS.shift(2)) & (lS.shift(2) < lS.shift(3))
  )
  isUp3S = (
    (hS > hS.shift(1)) & (hS.shift(1) > hS.shift(2)) & (hS.shift(2) > hS.shift(3))
    & (lS > lS.shift(1)) & (lS.shift(1) > lS.shift(2)) & (lS.shift(2) > lS.shift(3))
  )
  isEntryS = isDown3S & (ratio250S > 1) & (ratio5S < 1)
  isExitS = ratio5S > 1
  isEntryShortS = isUp3S & (ratio250S < 1) & (ratio5S > 1)
  isExitShortS = ratio5S < 1
  dw[und] = (
    ql.getStateS_timestop(isEntryS, isExitS, 9, isCleaned=False, isMonthlyRebal=False)
    - 2 * ql.getStateS_timestop(isEntryShortS, isExitShortS, 9, isCleaned=False, isMonthlyRebal=False)
  )
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['dfDict'] = dfDict
  d['ratio5S'] = ratio5S
  d['ratio250S'] = ratio250S
  return d

def runSTK(yrStart, isSkipTitle=False):
  script = 'STK'
  if not isSkipTitle:
    st.header(script)
  d = runSTKCore(yrStart)
  st.header('Table')
  df = d['dfDict']['SPY']
  tableS = ul.merge(
    df['Close'].round(2), df['High'].round(2), df['Low'].round(2),
    d['ratio5S'].round(3), d['ratio250S'].round(3), how='inner')
  ql.stWriteDf(tableS.tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runYPSCore(yrStart):
  # XLY/XLP 1d ratio thrust > 1.75 x sd60 -> long SPY (ES) same close; TS5
  # VT 0.29 / maxWgt 1.5 — ql.bt MDD 14.9% since 2017 (2020-03-16; .30 = 15.4%) (2026-11-11)
  # Seated @b 2026-11-12 (swap WRS→YPS; ART 83.3 Strong).
  und = 'SPY'
  etc = ['XLY', 'XLP']
  volTgt = .29
  maxWgt = 1.5
  dp, dw, dfDict, hv = ql.btSetup([und] + etc, yrStart=yrStart - 1)
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
    hv = hv.drop(und2, axis=1)
  #####
  yS = dfDict['XLY']['Close']
  pS = dfDict['XLP']['Close']
  relS = (np.log(yS).diff() - np.log(pS).diff()).rename('Rel')
  thrS = (1.75 * relS.rolling(60).std().shift(1)).rename('Thr')
  chgS = (yS / pS).pct_change().rename('XLY/XLP 1D')
  isEntryS = chgS > thrS
  isExitS = isEntryS & False
  stateS = ql.getStateS_timestop(isEntryS, isExitS, 5, isCleaned=True, isMonthlyRebal=True)
  dw[und] = stateS
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['chgS'] = chgS
  d['thrS'] = thrS
  return d

def runYPS(yrStart, isSkipTitle=False):
  script = 'YPS'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runYPSCore(yrStart)
  st.header('Table')
  ql.stWriteDf(ul.merge(d['chgS'].round(3), d['thrS'].round(3), how='inner').tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

#####
# Other MRs
#####
def runCOS2Core(yrStart):
  # SMH Ret1D<-2% ∩ ratio>1 → SOXL close→next open (all sessions).
  # Fixed size 1/3 (3× und ≈ 1× notionally). Seated @b 2026-11-19.
  # Dict mirrors COS (+ ret1S, ratioS). Signal SMH; trade SOXL.
  und, size = 'SOXL', 1 / 3
  etc = ['SMH']
  _, _, dfDict, _ = ql.btSetup([und] + etc, yrStart=yrStart - 1)
  #####
  soxlDf = dfDict[und]
  smhDf = dfDict['SMH']
  oS, cS = soxlDf['Open'], soxlDf['Close']
  ret1S = ql.applyDates(smhDf['Close'].pct_change(), cS).rename('SMH Ret 1D')
  ratioS = ql.applyDates(
    smhDf['Close'] / smhDf['Close'].rolling(200).mean(), cS).rename('Ratio')
  stateS = ((ret1S < -0.02) & (ratioS > 1)).fillna(False).rename('State')
  df = ul.merge(
    ql.applyDates(smhDf['Close'], cS).rename('SMH Close'),
    oS.rename('SOXL Open'),
    cS.rename('SOXL Close'),
    how='inner')
  #####
  openIdx = soxlDf.index + pd.Timedelta(hours=9, minutes=30)
  closeIdx = soxlDf.index + pd.Timedelta(hours=16)
  px = pd.concat([
    pd.Series(oS.values, index=openIdx, name=und),
    pd.Series(cS.values, index=closeIdx, name=und),
  ]).sort_index().to_frame()
  dwI = pd.DataFrame(0.0, index=px.index, columns=[und])
  dwI.loc[closeIdx[stateS.values], und] = size
  dwI.loc[dwI.index.year < yrStart] = 0
  #####
  r = (stateS.astype(float).shift(1) * size * (oS / cS.shift(1) - 1)).fillna(0)
  r.loc[r.index.year < yrStart] = 0
  dp = (1 + r).cumprod().rename('COS2').to_frame()
  dw = dp * np.nan
  dw.iloc[ql.endpoints(dw), 0] = 1
  #####
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['dpIntraday'] = px
  d['dwIntraday'] = dwI
  d['df'] = df
  d['ret1S'] = ret1S
  d['ratioS'] = ratioS
  d['stateS'] = stateS.astype(float)
  return d

def runCOS2(yrStart, isSkipTitle=False):
  script = 'COS2'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runCOS2Core(yrStart)
  st.header('Table')
  tableS = ul.merge(
    d['df']['SMH Close'].round(2),
    d['df']['SOXL Open'].round(2),
    d['df']['SOXL Close'].round(2),
    d['ret1S'].round(3), d['ratioS'].round(3), d['stateS'], how='inner')
  ql.stWriteDf(tableS.tail())
  st.header('Weights')
  ql.dwTail(d['dwIntraday'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runHMRCore(yrStart):
  # 2828.HK: IBS<0.20 & IBS2<0.4 & WPR14<-70; exit IBS>0.50 or 4d; VT 0.32 / maxWgt 1
  # (vs fixed-1: 2017 nearly flat; 2008 DD 32.6%→18.9%). Prior 0.36/1.5 sat on 1.5 cap.
  und = '2828.HK'
  volTgt = .32
  maxWgt = 1.0
  dp, dw, dfDict, hv = ql.btSetup([und], yrStart=yrStart - 1)
  df = dfDict[und]
  ibsS = ql.getIbsS(df)
  ibs2S = ql.getIbsS(df, n=2).rename('IBS2')
  wprS = ta.willr(df['High'], df['Low'], df['Close'], length=14).rename('WPR')
  isEntryS = (ibsS < .20) & (ibs2S < .4) & (wprS < -70)
  isExitS = ibsS > .50
  stateS = ql.getStateS_timestop(isEntryS, isExitS, 4, isCleaned=True, isMonthlyRebal=True)
  dw[und] = stateS
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['dfDict'] = dfDict
  d['ibsS'] = ibsS
  d['ibs2S'] = ibs2S
  d['wprS'] = wprS
  return d

def runHMR(yrStart, isSkipTitle=False):
  script = 'HMR'
  if not isSkipTitle:
    st.header(script)
  d = runHMRCore(yrStart)
  st.header('Table')
  df = d['dfDict']['2828.HK']
  df2 = ul.merge(df['Close'].round(2), d['ibsS'].round(3), d['ibs2S'].round(3), d['wprS'].round(1), how='inner')
  ql.stWriteDf(df2.tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runJMRCore(yrStart):
  # EWJ/−FXY: DXJ IBS<.15 ∩ EWJ IBS<.3; exit DXJ IBS>.7 | TS7; VT 0.32 / maxWgt 1 on EWJ
  # (was fixed ±1 — HV usually <0.32 so clip binds; align with HMR size)
  volTgt = .32
  maxWgt = 1.0
  etc=['DXJ']
  dp, dw, dfDict, hv = ql.btSetup(ul.spl('EWJ,FXY')+etc,yrStart=yrStart-1)
  dp2 = dp.copy()
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
    hv = hv.drop(und2, axis=1)
  #####
  ibs_DXJ = ql.getIbsS(dfDict['DXJ']).rename('IBS DXJ')
  ibs_EWJ   = ql.getIbsS(dfDict['EWJ']).rename('IBS EWJ')
  isEntryS = (ibs_DXJ < .15) & (ibs_EWJ < .3)
  isExitS  = ibs_DXJ > .7
  stateS   = ql.getStateS_timestop(isEntryS, isExitS, 7, isCleaned=True, isMonthlyRebal=True)
  #####
  dw['EWJ'] = stateS
  dw['EWJ'] = (dw['EWJ'] * volTgt / hv['EWJ']).clip(0, maxWgt)
  dw['FXY'] = -dw['EWJ']
  #####
  d=dict()
  d['dp']=dp
  d['dp2']=dp2
  d['dw']=dw
  d['dfDict']=dfDict
  d['ibs_DXJ']=ibs_DXJ
  d['ibs_EWJ']=ibs_EWJ
  return d

def runJMR(yrStart,isSkipTitle=False):
  script = 'JMR'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runJMRCore(yrStart)
  st.header('Table')
  df2 = ul.merge(d['dp2'],d['ibs_DXJ'].round(3), d['ibs_EWJ'].round(3), how='inner')
  ql.stWriteDf(df2.tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

#####
# Hedges
#####
def runSCI(yrStart,isSkipTitle=False):
  script = 'SCI'
  if not isSkipTitle:
    st.header(script)
  ######
  dp, dw, dfDict, hv = ql.btSetup(ul.spl('IEI,REM,GLD,GDXJ,XLE,OIH'),yrStart=yrStart-1)
  idx = dw.index[ql.endpoints(dp)]
  dw.loc[idx, 'IEI'] = 0.8
  dw.loc[idx, 'REM'] = -0.4
  dw.loc[idx, 'GLD'] = 0.6
  dw.loc[idx, 'GDXJ'] = -0.2
  dw.loc[idx, 'XLE'] = 0.6
  dw.loc[idx, 'OIH'] = -0.4
  st.header('Prices')
  ql.stWriteDf(dp.tail())
  st.header('Weights')
  ql.dwTail(dw)
  ql.bt(script, dp, dw, yrStart)


def runVCACore(yrStart):
  und='VIXM'
  etc=ul.spl('SPY')
  dp, dw, dfDict, _ = ql.btSetup([und]+etc,yrStart=yrStart-1)
  spyS = (dfDict['SPY']['Close']).rename('SPY')
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
  #####
  spyRatioS = (spyS / spyS.rolling(200).mean()).rename('SPY Ratio')
  ibsS = ql.getIbsS(dfDict['SPY'])
  #####
  vixS = ql.applyDates(pl.getPriceHistory('VIX.INDX', yrStart=yrStart-1)['Close'], spyS).rename('VIX')
  vixRatioS = (vixS / vixS.rolling(10).mean()).rename('VIX Ratio')
  hvS = (spyS.pct_change().rolling(10).std() * math.sqrt(252) * 100).rename('HV')
  eVRPS= (vixS-hvS).rename('eVRPS')
  eVRPS_pctl = eVRPS.rolling(252).rank(pct=True).rename('eVRPS Pctl')
  #####
  m= lambda s: ql.applyDates(s,dw).ffill().fillna(0)
  w1 = m((spyRatioS < 1) & (ibsS > 0.75) & (vixRatioS > 1))
  w2 = m((eVRPS_pctl <= 0.25) & (vixRatioS > 1))
  dw[und] = ql.cleanS((w1 + w2).clip(upper=1), isMonthlyRebal=False)
  dw=ql.cleanS(dw,isMonthlyRebal=True)
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
  ql.stWriteDf(tableS.tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runVCA2Core(yrStart):
  und='VIXY'
  etc=ul.spl('SPY')
  dp, dw, dfDict, _ = ql.btSetup([und]+etc,yrStart=yrStart-1)
  spyS = (dfDict['SPY']['Close']).rename('SPY')
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
  #####
  vix1DS = ql.applyDates(pl.getPriceHistory('VIX1D.INDX', yrStart=yrStart-1)['Close'], spyS).rename('VIX1D')
  sma3S = vix1DS.rolling(3).mean().rename('VIX1D SMA3')
  isAthS = (spyS == spyS.rolling(252).max()).rename('SPY ATH?')*1
  #####
  isEntryS = (sma3S <= 10) & (isAthS == 1)
  isExitS = sma3S > 12
  dw[und] = ql.getStateS(isEntryS, isExitS, isCleaned=True, isMonthlyRebal=True)
  #####
  d=dict()
  d['dp']=dp
  d['dw']=dw
  d['vix1DS'] = vix1DS
  d['spyS'] = spyS
  d['sma3S'] = sma3S
  d['isAthS'] = isAthS
  return d

def runVCA2(yrStart,isSkipTitle=False):
  script = 'VCA2'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runVCA2Core(yrStart)
  st.header('Tables')
  tableS = ul.merge(d['dp'], d['vix1DS'].round(2), d['sma3S'].round(2),
                    d['spyS'],d['isAthS'].astype(int), how='inner')
  ql.stWriteDf(tableS.tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

#####
# VCA3 — gold ATH ∩ VIX9D SMA3≤14 (Lab 2026-11-01); seated @b
# ATH ∩ GLD↑ ∩ GLD/SMA50>1 ∩ VIX9D SMA3≤14 → VIXY hold1d
#####
def runVCA3Core(yrStart):
  und='VIXY'
  etc=ul.spl('SPY,GLD')
  dp, dw, dfDict, _ = ql.btSetup([und]+etc,yrStart=yrStart-1)
  spyS = (dfDict['SPY']['Close']).rename('SPY')
  gldS = (dfDict['GLD']['Close']).rename('GLD')
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
  #####
  isAthS = (spyS == spyS.rolling(252).max()).rename('SPY ATH?')*1
  isGLDUpS = (gldS.pct_change() > 0).rename('GLD↑?')*1
  ratioS = (gldS / gldS.rolling(50).mean()).rename('Ratio')
  vix9DS = ql.applyDates(pl.getPriceHistory('VIX9D.INDX', yrStart=yrStart-1)['Close'], spyS).rename('VIX9D')
  sma3S = vix9DS.rolling(3).mean().rename('VIX9D SMA3')
  #####
  dw[und] = ql.cleanS(
    ((isAthS == 1) & (isGLDUpS == 1) & (ratioS > 1) & (sma3S <= 14))*1,
    isMonthlyRebal=True,
  )
  #####
  d=dict()
  d['dp']=dp
  d['dw']=dw
  d['spyS']=spyS
  d['gldS']=gldS
  d['isAthS']=isAthS
  d['isGLDUpS']=isGLDUpS
  d['ratioS']=ratioS
  d['vix9DS']=vix9DS
  d['sma3S']=sma3S
  return d

def runVCA3(yrStart,isSkipTitle=False):
  script = 'VCA3'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runVCA3Core(yrStart)
  st.header('Tables')
  tableS = ul.merge(d['dp'], d['spyS'], d['gldS'].round(2),
                    d['isAthS'].astype(int), d['isGLDUpS'].astype(int),
                    d['ratioS'].round(3), d['vix9DS'].round(2),
                    d['sma3S'].round(2), how='inner')
  ql.stWriteDf(tableS.tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

#####
# VCA4 — SPY↑ ∩ VVIX day-chg>2% ∩ VVIX/SMA20>1 → VIXY (Lab 2026-11-18)
# Fear-on-rally / vol-of-vol; @b with BCS a→b. Ex-SVH rename.
#####
def runVCA4Core(yrStart):
  und = 'VIXY'
  etc = ul.spl('SPY')
  dp, dw, dfDict, _ = ql.btSetup([und] + etc, yrStart=yrStart - 1)
  spyS = (dfDict['SPY']['Close']).rename('SPY')
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
  #####
  vvixS = ql.applyDates(pl.getPriceHistory('VVIX.INDX', yrStart=yrStart - 1)['Close'], spyS).rename('VVIX')
  vvixChgS = vvixS.pct_change().rename('VVIX Chg%')
  ratioS = (vvixS / vvixS.rolling(20).mean()).rename('Ratio')
  #####
  dw[und] = ql.cleanS(
    ((spyS.pct_change() > 0) & (vvixChgS > .02) & (ratioS > 1)) * 1,
    isMonthlyRebal=True,
  )
  #####
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['spyS'] = spyS
  d['vvixS'] = vvixS
  d['vvixChgS'] = vvixChgS
  d['ratioS'] = ratioS
  return d

def runVCA4(yrStart, isSkipTitle=False):
  script = 'VCA4'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runVCA4Core(yrStart)
  st.header('Tables')
  tableS = ul.merge(
    d['dp'], d['spyS'], d['vvixS'].round(2), d['vvixChgS'].round(3),
    d['ratioS'].round(3), how='inner')
  ql.stWriteDf(tableS.tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

#####
# Alt1
#####
def runBCSCore(yrStart):
  # BTC trend: >50DMA & >0.8x240d high; monthly; VT 0.20 / maxWgt 1 (Tier-2 und; Harness 2026-11-18)
  # Renamed BTS→BCS 2026-11-15.
  volTgt = .20
  maxWgt = 1.0
  cS = pl.getPriceHistoryCrypto('BTC', yrStart=yrStart)['Close']
  ratioS = (cS / cS.rolling(50).mean()).rename('Ratio')
  ratio2S = (cS / cS.rolling(240).max()).rename('Ratio2')
  dw = ((ratioS > 1) & (ratio2S > 0.8)).rename('BTC').to_frame()
  dp = cS.rename('BTC').to_frame()
  hv = ql.getHV(dp, af=365)
  dw = ql.cleanS(dw, isMonthlyRebal=True)
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['ratioS'] = ratioS
  d['ratio2S'] = ratio2S
  return d

def runBCS(yrStart, isSkipTitle=False):
  script = 'BCS'
  if not isSkipTitle:
    st.header(script)
  d = runBCSCore(yrStart)
  st.header('Table')
  tableS = ul.merge(d['dp'], d['ratioS'].round(3), d['ratio2S'].round(3), how='inner')
  ql.stWriteDf(tableS.tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runNGSCore(yrStart):
  # UNG idio down SigmaSpike (TS3) OR session-before-EIA WNGSR Nov–Mar (TS1)
  # Dense timestop per leg → OR → one ql.cleanS (do not clean before OR).
  # VT 0.32 / maxWgt 1 — was 0.24 (OR MDD-edge); raised 2026-11-18 (2017+ MDD~13.6 PASS).
  und = 'UNG'
  volTgt = .32
  maxWgt = 1.0
  dp, dw, dfDict, hv = ql.btSetup([und, 'SPY'], yrStart=yrStart - 1)
  dp = dp[[und]]; dw = dw[[und]]; hv = hv[[und]]
  #####
  cS = dfDict[und]['Close']
  zUNGS = ql.getSigmaSpikeS(cS, prefix='UNG')
  zSPYS = ql.getSigmaSpikeS(dfDict['SPY']['Close'], prefix='SPY')
  ratioS = (cS / cS.rolling(200).mean()).rename('Ratio')
  isEntryAS = ql.applyDatesBoolean((zUNGS < -1.5) & ~(zSPYS < -1) & (ratioS < 1), dw)
  isPreEiaS = ql.getIsPreEiaEntryS(dw.index)  # last session before WNGSR (Thu default + holidays)
  isWinterS = pd.Series(dw.index.month, index=dw.index).isin([11, 12, 1, 2, 3])
  isEntryBS = (isPreEiaS & isWinterS).fillna(False)
  stateS = (
    (ql.getStateS_timestop(isEntryAS, isEntryAS & False, 3) > 0)
    | (ql.getStateS_timestop(isEntryBS, isEntryBS & False, 1) > 0)
  ).astype(float)
  stateS = ql.cleanS(stateS, isMonthlyRebal=True)
  dw[und] = -stateS
  dw = (dw * volTgt / hv).clip(-maxWgt, 0)
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['zUNGS'] = zUNGS
  d['zSPYS'] = zSPYS
  d['ratioS'] = ratioS
  d['isPreEiaS'] = isPreEiaS
  return d

def runNGS(yrStart, isSkipTitle=False):
  script = 'NGS'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runNGSCore(yrStart)
  st.header('Table')
  ql.stWriteDf(ul.merge(d['dp'], d['zUNGS'].round(2), d['zSPYS'].round(2),
                     d['ratioS'].round(3), d['isPreEiaS'].astype(int),
                     how='inner').tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runSISCore(yrStart):
  # SLV SigmaSpike (TS3) OR ChinaLongHoliday (XSHG gap≥5; L→F NYSE closes, nPost=1)
  # Dense per leg → OR → one ql.cleanS. VT 0.32 / maxWgt 1. Seated @b Alt1.
  # Spike freeze 2026-11-14; ChinaLongHoliday merge 2026-11-16 (nPost=1 exit).
  und = 'SLV'
  volTgt = .32
  maxWgt = 1.0
  dp, dw, dfDict, hv = ql.btSetup([und], yrStart=yrStart - 1)
  #####
  cS = dfDict[und]['Close']
  spikeS = ql.getSigmaSpikeS(cS, prefix='SLV')
  isEntryAS = ql.applyDatesBoolean(spikeS > 1.5, dw)
  isChinaLongHolidayS = ql.getChinaLongHolidayDenseS(dw.index)
  stateS = (
    (ql.getStateS_timestop(isEntryAS, isEntryAS & False, 3) > 0)
    | isChinaLongHolidayS
  ).astype(float)
  stateS = ql.cleanS(stateS, isMonthlyRebal=True)
  dw[und] = stateS
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['spikeS'] = spikeS
  d['isChinaLongHolidayS'] = isChinaLongHolidayS
  return d

def runSIS(yrStart, isSkipTitle=False):
  script = 'SIS'
  if not isSkipTitle:
    st.header(script)
  #####
  d = runSISCore(yrStart)
  st.header('Table')
  ql.stWriteDf(ul.merge(
    d['dp'], d['spikeS'].round(2), d['isChinaLongHolidayS'].astype(int),
    how='inner'
  ).tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runWTSCore(yrStart):
  # WEAT L/S: ratio10/50 + quiet3; next-open; VT 0.17 / maxWgt 1 (niche rulebook)
  volTgt, maxWgt, quietN = .17, 1.0, 3
  und = 'WEAT'
  dp, dw, dfDict, hv = ql.btSetup([und], yrStart=yrStart - 1)
  oS = dfDict[und]['Open']
  cS = dfDict[und]['Close']
  ratio10S = (cS / cS.shift(10)).rename('Ratio 10D')
  ratio50S = (cS / cS.shift(50)).rename('Ratio 50D')
  isLongS = ((ratio10S > 1) & (ratio50S < 1)).fillna(False)
  isShortS = ((ratio10S < 1) & (ratio50S > 1)).fillna(False)
  isQuiet3S = ((~isLongS & ~isShortS).rolling(quietN).sum() >= quietN).fillna(False)
  #####
  dp[und] = oS.shift(-1)
  dp.loc[dp.index[-1], und] = dp.loc[dp.index[-2], und]
  dw[und] = ql.getStateS(isLongS, isShortS | isQuiet3S) - ql.getStateS(isShortS, isLongS | isQuiet3S)
  dw = ql.cleanS(dw, isMonthlyRebal=True)
  dw = dw * (volTgt / hv).clip(0, maxWgt)
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['ratio10S'] = ratio10S
  d['ratio50S'] = ratio50S
  return d

def runWTS(yrStart, isSkipTitle=False):
  script = 'WTS'
  if not isSkipTitle:
    st.header(script)
  d = runWTSCore(yrStart)
  st.header('Table')
  ql.stWriteDf(ul.merge(d['dp'], d['ratio10S'].round(3), d['ratio50S'].round(3), how='inner').tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

#####
# Etc
#####
# AHS — Investlab https://investlab.substack.com/p/adaptive-hedge-a-cheaper-way-to-survive
#        Carlson   https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5334772
#        ICOM.LSE listed 2017-07-26; PriceLib switches from DJP splice on 2018-10-31
def runAHSCore(yrStart):
  # 2026-10-21: drop SH only; keep GLD; VT raised to size-limits MDD edge (<15%)
  # 2026-11-18: month-flat when isGLDNearHHV63S (TAA GLD-booth cousin abrasion)
  volTgt, maxWgt = .18, 1.5
  unds = ul.spl('TLT,ICOM.LSE,GLD,UUP')
  lbs = (3, 6, 12)
  dp, dw, dfDict, hv = ql.btSetup(unds, yrStart=yrStart - 1)
  #####
  meIdx = dp.index.to_series().groupby(dp.index.to_period('M')).max()
  me = dp.loc[meIdx]
  scoreS = sum(me.pct_change(k) for k in lbs) / len(lbs)
  nposS = sum((me.pct_change(k) > 0) * 1 for k in lbs)
  ratioS = dp / dp.rolling(200).mean()
  isGLDNearHHV63S = ql.applyDatesBoolean(ql.getIsGLDNearHHV63S(dfDict['GLD'], thr=.98), dw) * 1.0
  #####
  held = []
  for dt in me.index:
    sc = scoreS.loc[dt]
    if sc.isna().any():
      continue
    if float(isGLDNearHHV63S.loc[dt]) > 0:
      dw.loc[dt] = 0.0
      held = []
      continue
    npos = nposS.loc[dt].where(ratioS.loc[dt] > 1, 0)
    ranked = sc.sort_values(ascending=False).index
    rank = pd.Series(range(1, len(ranked) + 1), index=ranked)
    nxt = [t for t in held if rank[t] <= 3 and npos[t] >= 2]
    for t in ranked:
      if t in nxt:
        continue
      if rank[t] <= 2 and npos[t] >= 2:
        nxt.append(t)
      if len(nxt) >= 2:
        break
    nxt = nxt[:2]
    held = nxt
    raw = pd.Series(0.0, index=unds)
    raw[nxt] = .5
    dw.loc[dt] = (raw * volTgt / hv.loc[dt]).clip(0, maxWgt)
  #####
  d = dict()
  d['dp'] = dp
  d['dw'] = dw
  d['ratioS'] = ratioS
  d['isGLDNearHHV63S'] = isGLDNearHHV63S
  return d

def runAHS(yrStart, isSkipTitle=False):
  script = 'AHS'
  if not isSkipTitle:
    st.header(script)
  d = runAHSCore(yrStart)
  st.header('Prices')
  ql.stWriteDf(d['dp'].tail())
  st.header('Table')
  ql.stWriteDf(ul.merge(
    d['ratioS'].round(3), d['isGLDNearHHV63S'], how='inner').tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

def runGEOCore(yrStart):
  volTgt = .32
  #####
  wDict = {'NATO.LSE': .5, 'NUCL.LSE': .5}
  unds = list(wDict.keys())
  etc = ul.spl('ITA,U-UN.TO')
  dp, dw, dfDict, hv = ql.btSetup(unds + etc, yrStart=yrStart - 1)
  dp2 = dp.copy()
  for und2 in etc:
    dp = dp.drop(und2, axis=1)
    dw = dw.drop(und2, axis=1)
    hv = hv.drop(und2, axis=1)
  #####
  # NATO.LSE signal
  itaS = dfDict['ITA']['Close']
  itaMS = itaS.iloc[ql.endpoints(itaS)]
  ratio12S_ITA = (itaMS / itaMS.rolling(12).mean()).rename('ITA Ratio 12M')
  dw['NATO.LSE'] = ql.applyDates(ratio12S_ITA > 1, dw)
  #####
  # NUCL.LSE signal
  uunS = dfDict['U-UN.TO']['Close']
  uunMS = uunS.iloc[ql.endpoints(uunS)]
  rocS_UUN = uunMS.ffill().pct_change(fill_method=None).rename('UUN ROC 1M')
  dw['NUCL.LSE'] = ql.applyDates(rocS_UUN > 0, dw)
  #####
  dw = ql.cleanS(dw, isMonthlyRebal=True)
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
  ql.stWriteDf(ul.merge(d['dp2'], d['ratio12S_ITA'].round(3), d['rocS_UUN'].round(3), how='inner').tail())
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)


def runZBTCore(yrStart):
  # Carson / Detrick official ZBT dates (table 25 Apr 2025)
  # 2026-11-19: VT0.32 / maxWgt1.5 (Simon; dig preferred fixed2 — confidence cut).
  zbtDates = [
    '2009-03-18','2011-10-14','2013-10-08','2015-10-08','2019-01-07','2023-03-31',
    '2023-11-03','2025-04-25',
  ]
  und = 'QQQ'
  volTgt = .32
  maxWgt = 1.5
  dp, dw, dfDict, hv = ql.btSetup([und],yrStart=yrStart-1)
  #####
  eom = ql.getNYSEEomS(dp.index).fillna(0) == 1
  sigM = pd.to_datetime(zbtDates).to_period('M')
  m = dp.index.to_period('M')
  isEntryS = eom & m.isin(sigM)
  isExitS = eom & m.isin(sigM + 3) & ~isEntryS
  dw[und] = ql.getStateS(isEntryS, isExitS, isCleaned=True, isMonthlyRebal=True)
  dw = (dw * volTgt / hv).clip(0, maxWgt)
  d=dict()
  d['dp']=dp
  d['dw']=dw
  return d

def runZBT(yrStart,isSkipTitle=False):
  script = 'ZBT'
  if not isSkipTitle:
    st.header(script)
  #####
  d=runZBTCore(yrStart)
  st.header('Prices')
  ql.dwTail(d['dp'])
  st.header('Weights')
  ql.dwTail(d['dw'])
  ql.bt(script, d['dp'], d['dw'], yrStart)

#####


#####

def runAggregate(yrStart,strategies,weights,script,isBFill=False, isCorrs=False):
  st.header(script)
  #####
  # Weights
  st.header('Weights')
  z = zip(strategies, weights)
  df = pd.DataFrame(z, columns=ul.spl('Strategy,Weight')).set_index('Strategy')
  ql.stWriteDf(df)
  #####
  # Calcs
  dp = pd.DataFrame()
  for strategy in strategies:
    dp[strategy] = ul.cachePersist('r', strategy)
  dp = ql.applyDates(dp, dp.iloc[:,-1]).ffill()
  if isBFill: dp=dp.bfill()
  dw = dp * np.nan
  pe = ql.endpoints(dw)
  for i in range(len(weights)):
    dw.iloc[pe, i] = weights[i]
  #####
  # Backtest
  ql.bt(script, dp, dw, yrStart)
  #####
  # Corrs
  if isCorrs:
    st.header('Corrs')
    ql.stWriteDf(dp.pct_change().corr().round(3))
  #####
  # Recent performance
  st.header('Recent Performance')
  dp2 = dp.copy()
  dp2[script] = ul.cachePersist('r', script)
  dp2 = dp2[[script] + strategies]
  dp2 = (dp2 / dp2.iloc[-1]).tail(23) * 100
  dp2 = dp2.round(2)
  ql.stWriteDf(dp2, isMaxHeight=True)
