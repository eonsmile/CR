###############
# Quant Library
###############
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
    # d['mar'] = d['cagr']/d['maxdd']  # old: MAR = CAGR/MaxDD; Calmar = same on trailing 3y
    # Ulcer Index = RMS of daily % underwater; UPI = CAGR/Ulcer (aka Martin / Ulcer Performance Index)
    d['ulcer'] = (dd.pow(2).mean()) ** 0.5
    d['upi'] = d['cagr'] / d['ulcer']
    return d
  #####
  d=m(ecS)
  d3=m(ecS[ecS.index>pendulum.instance(ecS.index[-1]).subtract(years=3).naive()])
  #####
  m=lambda label,z: f"{label}: <font color='red'>{z}</font>"
  sep='&nbsp;'*10
  st.markdown(sep.join([
    # old: m('&nbsp;' * 3 + 'Calmar', f"{d3['mar']:.2f}"),
    # old: m('MAR', f"{d['mar']:.2f}"),
    m('&nbsp;' * 3 + 'UPI', f"{d['upi']:.2f}"),
    m('UPI3y', f"{d3['upi']:.2f}"),
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
  r = s.ffill().pct_change(fill_method=None)[1:]
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

def applyDatesBoolean(a, b):
  # Align bool/flag series to b's index; pre-history → False (never NaN).
  # nullable BooleanDtype avoids FutureWarning on fillna downcast
  return applyDates(a, b).astype('boolean').fillna(False).astype(bool)

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

def getChinaLongHolidayDenseS(idx, minGapDays=5):
  """China long-holiday dense on/off series on the NYSE index (SIS leg).

  Dense = one bool per bar (in or out every day), not a sparse list of
  entry/exit dates. Long holiday = two consecutive XSHG sessions
  ≥ minGapDays calendar days apart (weekend = 3; one-day holiday beside a
  weekend = 4). L = last XSHG session before the gap; F = first after.
  Long SLV from NYSE close on/after L through the bar before NYSE close
  on/after F (nPost=1).

  Look-ahead: XSHG/NYSE calendars are loaded past idx[-1] (and
  XSHG_EXTRA_CLOSURES for years the library has not baked in), so a gap is
  known on L — and earlier once the State Council override list includes that
  year — without waiting for future SLV bars. pandas_market_calendars does not
  auto-update 2027+ China holidays; extend XSHG_EXTRA_CLOSURES +
  XSHG_VERIFIED_THRU each Nov/Dec from the gov.cn notice or the sleeve errors
  past the verified year.
  """
  idxIn = pd.DatetimeIndex(idx)
  norm = idxIn.tz_localize(None) if idxIn.tz is not None else idxIn
  norm = norm.normalize()
  if int(norm[-1].year) > XSHG_VERIFIED_THRU:
    ul.iExit(
      f'getChinaLongHolidayDenseS: XSHG calendar not verified past '
      f'{XSHG_VERIFIED_THRU}; add State Council closures and bump '
      f'XSHG_VERIFIED_THRU'
    )
  start = (norm[0] - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
  # Past last tape bar so upcoming L/F are visible mid-approach / mid-holiday.
  end = (max(norm[-1], pd.Timestamp(f'{XSHG_VERIFIED_THRU}-12-31'))
         + pd.Timedelta(days=400)).strftime('%Y-%m-%d')
  nyse = pd.DatetimeIndex(
    pandas_market_calendars.get_calendar('NYSE').valid_days(
      start_date=start, end_date=end
    )
  ).tz_localize(None)
  out = pd.Series(False, index=idxIn, name='ChinaLongHoliday?')
  by_day = {}
  for ts, day in zip(idxIn, norm):
    by_day.setdefault(pd.Timestamp(day).normalize(), ts)
  for L, F in getXshgClosures(start, end, minGapDays):
    iIn = int(nyse.searchsorted(L))
    iOut = int(nyse.searchsorted(F))
    if iIn >= len(nyse) or iOut >= len(nyse):
      continue
    dIn, dOut = nyse[iIn], nyse[iOut]
    if dOut <= dIn:
      continue
    for day in nyse[(nyse >= dIn) & (nyse < dOut)]:
      key = pd.Timestamp(day).normalize()
      if key in by_day:
        out.loc[by_day[key]] = True
  return out

def getEiaWngsrReleaseDates(start, end):
  """WNGSR release dates from calendars/eia_wngsr.txt (checked-in SoT).

  Append one ISO date per line when EIA posts new schedule rows
  (https://ir.eia.gov/ngs/schedule.html). Past the last pin date, falls back to
  the federal-holiday rule (not NYSE): Thu holiday → Wed; Tue holiday → Fri;
  Wed/Mon alone stay Thu. Pin file is the SoT for history through ~2027.
  """
  import re
  from pandas.tseries.holiday import USFederalHolidayCalendar
  start, end = pd.Timestamp(start), pd.Timestamp(end)
  start_d, end_d = start.date(), end.date()
  pin_path = Path(__file__).resolve().parent / 'calendars' / 'eia_wngsr.txt'
  pin = []
  if pin_path.exists():
    for line in pin_path.read_text(encoding='utf-8').splitlines():
      s = line.strip()
      if not s or s.startswith('#'):
        continue
      m = re.match(r'^(\d{4}-\d{2}-\d{2})$', s)
      if m:
        pin.append(pd.Timestamp(m.group(1)).date())
  pin = sorted(set(pin))
  out = [d for d in pin if start_d <= d <= end_d]
  last_pin = pin[-1] if pin else None
  if last_pin is not None and end_d <= last_pin:
    return out

  # look-ahead past pin: federal holiday rule (matches pin rebuild)
  rule_start = pd.Timestamp(last_pin) + pd.Timedelta(days=1) if last_pin else start
  if rule_start > end:
    return out
  hol = set(pd.Timestamp(x).date() for x in USFederalHolidayCalendar().holidays(
    start=rule_start, end=end))
  hol.add(pd.Timestamp('2025-01-09').date())  # National Day of Mourning
  by_thu = {}
  for t in pd.date_range(rule_start, end, freq='D'):
    if t.dayofweek != 3:
      continue
    mon = t - pd.Timedelta(days=t.dayofweek)
    week = [(mon + pd.Timedelta(days=i)).date() for i in range(5)]
    tue, thu = week[1], week[3]
    fri = week[4]
    if thu in hol:
      rel = (t - pd.Timedelta(days=1)).date()
    elif tue in hol:
      rel = fri
    else:
      rel = thu
    if last_pin is not None and rel <= last_pin:
      continue
    if rel < start_d or rel > end_d:
      continue
    by_thu[thu] = rel
  out.extend(by_thu.values())
  return sorted(set(out))

def getIsPreEiaEntryS(idx):
  """True on the last NYSE session strictly before each WNGSR release date.

  Uses the NYSE calendar (not only printed idx) so look-ahead is correct when
  the tape has not yet reached the true entry day.
  """
  idxIn = pd.DatetimeIndex(idx)
  norm = idxIn.tz_localize(None) if idxIn.tz is not None else idxIn
  norm = norm.normalize()
  start = (norm[0] - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
  end = (norm[-1] + pd.Timedelta(days=21)).strftime('%Y-%m-%d')
  releases = getEiaWngsrReleaseDates(start, end)
  cal = pandas_market_calendars.get_calendar('NYSE')
  sessions = pd.DatetimeIndex([
    pd.Timestamp(x).date() for x in cal.valid_days(start_date=start, end_date=end)
  ])
  out = pd.Series(False, index=idxIn, name='Pre EIA?')
  # map normalized dates -> first matching label in idx (preserve original stamps)
  by_day = {}
  for ts, day in zip(idxIn, norm):
    by_day.setdefault(pd.Timestamp(day).date(), ts)
  for r in releases:
    pos = sessions.searchsorted(pd.Timestamp(r), side='left') - 1
    if pos < 0:
      continue
    entry = sessions[pos].date() if hasattr(sessions[pos], 'date') else pd.Timestamp(sessions[pos]).date()
    if entry in by_day:
      out.loc[by_day[entry]] = True
  return out

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

# XSHG long closures not yet in pandas_market_calendars (plain weekdays otherwise).
# Source: State Council holiday notices (gov.cn). Bump XSHG_VERIFIED_THRU when extended.
XSHG_EXTRA_CLOSURES = [  # (L last XSHG session before closure, F first session after)
  ('2027-02-04', '2027-02-15'),  # Spring Festival
  ('2027-04-30', '2027-05-06'),  # Labour Day
  ('2027-09-30', '2027-10-08'),  # National Day
]
XSHG_VERIFIED_THRU = 2027

def getXshgClosures(start, end, minGapDays=5):
  """(L, F) pairs where XSHG sessions are ≥ minGapDays calendar days apart.

  Uses pandas_market_calendars XSHG, then punches out XSHG_EXTRA_CLOSURES so
  years the library has not updated (e.g. 2027 LNY) still form a long gap.
  """
  cal = pandas_market_calendars.get_calendar('XSHG')
  sessions = pd.DatetimeIndex(
    cal.valid_days(start_date=start, end_date=end)
  ).tz_localize(None)
  for a, b in XSHG_EXTRA_CLOSURES:
    aT, bT = pd.Timestamp(a), pd.Timestamp(b)
    sessions = sessions[~((sessions > aT) & (sessions < bT))]
  if len(sessions) < 2:
    return []
  gapS = pd.Series(sessions[1:] - sessions[:-1], index=sessions[:-1])
  return [(a, a + g) for a, g in gapS[gapS >= pd.Timedelta(days=minGapDays)].items()]

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

def getIsGLDNearHHV63S(df, thr, lb=63):
  # True when GLD Close is within (1-thr) of the lb-day Highest High Value (HHV).
  # Shared by TAA (GLD booth neg voter) and AHS (month-flat gate) — callers must
  # pass thr (no default): TAA 0.96, AHS 0.98 as of 2026-11-18.
  # Raw bool (warm-up may be NaN); align with applyDatesBoolean.
  hiS = df['High'].rolling(int(lb)).max()
  return (df['Close'] >= hiS * float(thr)).rename('GLD Near HHV63?')

def getSigmaSpikeS(cS, n=60, isShift=True, prefix=''):
  # Adam Grimes SigmaSpike: r / rolling sd(n); default sd lagged 1 day
  rS = cS.astype(float).pct_change()
  sdS = rS.rolling(int(n)).std()
  if isShift:
    sdS = sdS.shift(1)
  name = f'{prefix} SigmaSpike' if prefix else 'SigmaSpike'
  return (rS / sdS).rename(name)

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
