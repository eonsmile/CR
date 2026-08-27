###############
# Price Library
###############
import UtilLib as ul
from UtilLib import SHARED_DICT
import streamlit as st
import numpy as np
import pandas as pd
import requests
import pendulum
import re
import yahooquery
import curl_cffi
import warnings
import os
import random
import time
import databento as db

#df2 = getPriceHistory('ITA', yrStart=yrStart)
#df2[['Close']].to_csv('tmp.csv', index_label='Date', date_format='%#m/%#d/%Y')

def getPriceHistory(und, yrStart=SHARED_DICT['yrStart']):
  dtStart=str(yrStart)+ '-1-1'
  if und.endswith('.T') or und.endswith('.TO'):
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
    return ul.extend(df, df2)

  ###########################################################################
  # DECOMMISSIONED underlyings: no longer used by alpha. CSVs live in
  # data/Archive/. Logic is identical to original except the CSV path.
  ###########################################################################
  if und in ul.spl('EUDF.XETRA,IPRE.XETRA,COM,INFL,IBIT,'
                   'DFNS.LSE,DRAM,ENCO.LSE,GCOW,HFGM,JEGA.LSE,ORR,PFIX,'
                   'RARE.LSE,ROBO,ROLL.LSE,TAIL,WCOA.LSE,'
                   'COPX,GRID,WTAI.LSE,REMX,9888.HK,9988.HK,DBMF,PFMN.TO'):
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
    elif und=='COPX':
      dtStart='2010-4-30'
    elif und=='GRID':
      dtStart='2009-11-30'
    elif und=='WTAI.LSE':
      dtStart = '2018-12-31'
    elif und=='REMX':
      dtStart='2010-10-29'
    elif und=='9888.HK':
      dtStart = '2021-3-23'
    elif und=='9988.HK':
      dtStart = '2019-11-26'
    elif und=='DBMF':
      dtStart = '2019-5-31'
    elif und == 'PFMN.TO':
      dtStart = '2019-7-31'
    else:
      dtStart = None
    if dtStart is not None: df = df.loc[df.index >= dtStart]
    df = m(df, f"{und}.csv", sub='Archive/')
    return df
  elif und == 'DFND.SW':
    return m(df, 'ITA.csv', sub='Archive/')
  elif und == 'BDRY':
    return m(df, 'BDI.csv', sub='Archive/')
  ###########################################################################
  # END decommissioned
  ###########################################################################

  if und in ul.spl('GDXJ,NATO.LSE,NUCL.LSE'):
    if und=='GDXJ':
      dtStart='2009-11-30'
    elif und == 'NATO.LSE':
      dtStart = '2023-7-31'
    elif und == 'NUCL.LSE':
      dtStart = '2023-2-28'
    else:
      dtStart = None
    if dtStart is not None: df = df.loc[df.index >= dtStart]
    df = m(df, f"{und}.csv")
  elif und=='VIX1D.INDX':
    dtStart = '2023-4-24'
    df = df.loc[df.index>=dtStart]
  elif und == '000660.KO':
    df = naverPatchHynix(df)
  return df

def getPriceHistoryCrypto(und, yrStart=SHARED_DICT['yrStart']):
  key = random.choice(ul.spl(os.getenv('CC_API_KEYS', '')))
  z = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={und}&tsym=USD&allData=true&api_key={key}"
  j = requests.get(z).json()
  data = j.get('Data')
  if not isinstance(data, dict) or 'Data' not in data:
    raise RuntimeError(f"cryptocompare histoday: {j.get('Message') or j}")
  df = pd.DataFrame(data['Data'])
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
'''


def getPriceHistoryDB(und, yrStart=SHARED_DICT['yrStart']):
  """Databento continuous futures OHLCV — same shape as getPriceHistory (EODHD).

  Root 'HG' → HG.v.0. Yearly parquet under data/databento/:
    - prior years: reuse cache if present, else pull once
    - current year: always live-pull (overwrite cache)
    - last year: re-pull unless file mtime is on/after Jan 15 of the current year
  Then ratio-adjust roll gaps. Auth: DB_API_KEY in .env.
  """

  # --- constants / helpers (local to this function) ---
  cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'databento')
  dataset = 'GLBX.MDP3'
  last_year_deadline_md = (1, 15)  # re-pull prior year unless mtime on/after Jan 15

  def _api_key():
    key = os.getenv('DB_API_KEY', '').strip()
    if not key:
      raise RuntimeError('DB_API_KEY missing from .env')
    return key

  def _normalize_symbol(nd_):
    s = str(nd_).strip()
    if re.fullmatch(r'[A-Za-z0-9]+\.[cnv]\.\d+', s):
      return s
    if re.fullmatch(r'[A-Za-z0-9]+', s):
      return f'{s}.v.0'
    return s

  def _year_path(safe_, year):
    return os.path.join(cache_dir, f'{safe_}_{year}.parquet')

  def _normalize_raw(df):
    if df is None or len(df) == 0:
      return None
    part = df.reset_index() if 'ts_event' not in getattr(df, 'columns', []) else df.copy()
    if 'ts_event' not in part.columns:
      part = part.reset_index()
    part['date'] = pd.to_datetime(part['ts_event']).dt.tz_convert(None).dt.normalize()
    part = part.sort_values('date').drop_duplicates('date', keep='last').set_index('date')
    return pd.DataFrame({
      'Open': part['open'].astype(float),
      'High': part['high'].astype(float),
      'Low': part['low'].astype(float),
      'Close': part['close'].astype(float),
      'Volume': part['volume'].astype(float),
      'instrument_id': part['instrument_id'].astype(int),
    })

  def _load_year(safe_, year):
    path = _year_path(safe_, year)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
      return None
    try:
      df = pd.read_parquet(path)
      if df is None or len(df) == 0:
        return None
      df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()
      return df.sort_index()
    except Exception:
      return None

  def _save_year(safe_, year, df):
    if df is None or len(df) == 0:
      return
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(_year_path(safe_, year))

  def _pull_year(client_, symbol_, year, y_now_, avail_end_, retries=4):
    start_y = f'{year}-01-01'
    end_y = f'{year + 1}-01-01' if year < y_now_ else avail_end_.strftime('%Y-%m-%dT%H:%M:%S')
    last_err = None
    for attempt in range(retries):
      try:
        data = client_.timeseries.get_range(
          dataset=dataset,
          schema='ohlcv-1d',
          symbols=symbol_,
          stype_in='continuous',
          start=start_y,
          end=end_y,
        )
        return _normalize_raw(data.to_df())
      except Exception as e:
        last_err = e
        msg = str(e)
        if 'data_start_before_available' in msg or 'dataset_unavailable_range' in msg:
          return None
        m_end = re.search(r"available_end['\":\s]+['\"]?([0-9T:\.\-\+Z]+)", msg)
        if m_end and attempt == 0:
          end_y = m_end.group(1).rstrip('Z').split('+')[0]
          continue
        time.sleep(1.2 * (attempt + 1))
    if last_err is not None:
      raise last_err
    return None

  def _need_refresh_last_year(safe_, last_y_, y_now_):
    path = _year_path(safe_, last_y_)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
      return True
    mtime = pd.Timestamp(os.path.getmtime(path), unit='s')
    deadline = pd.Timestamp(year=y_now_, month=last_year_deadline_md[0], day=last_year_deadline_md[1])
    return mtime < deadline

  def _ratio_adjust(part):
    close_raw = part['Close'].astype(float).values
    ids = part['instrument_id'].astype(int).values
    adj = {
      'Open': part['Open'].astype(float).values.copy(),
      'High': part['High'].astype(float).values.copy(),
      'Low': part['Low'].astype(float).values.copy(),
      'Close': close_raw.copy(),
    }
    for i in range(len(part) - 1, 0, -1):
      if ids[i] == ids[i - 1]:
        continue
      prev_px, cur_px = close_raw[i - 1], close_raw[i]
      if prev_px == 0 or np.isnan(prev_px) or np.isnan(cur_px):
        continue
      factor = cur_px / prev_px
      for a in adj.values():
        a[:i] *= factor
    out = pd.DataFrame({
      'Open': adj['Open'],
      'High': adj['High'],
      'Low': adj['Low'],
      'Close': adj['Close'],
      'Volume': part['Volume'].fillna(0).round(0).astype(np.int64).values,
    }, index=part.index)
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None).normalize()
    out.index.name = 'date'
    return out.sort_index().round(10)

  # --- main ---
  symbol = _normalize_symbol(und)
  safe = symbol.replace('.', '_')
  y0 = int(yrStart)

  client = db.Historical(_api_key())
  try:
    rng = client.metadata.get_dataset_range(dataset=dataset)
    avail = rng.get('schema', {}).get('ohlcv-1d', {}).get('end') or rng.get('end')
    avail_end = pd.Timestamp(str(avail).replace('Z', '').split('+')[0])
  except Exception:
    avail_end = pd.Timestamp.utcnow().normalize()
  y_now = int(avail_end.year)

  by_year = {}

  for year in range(y0, y_now):
    cached = _load_year(safe, year)
    if cached is not None:
      by_year[year] = cached
      continue
    fetched = _pull_year(client, symbol, year, y_now, avail_end)
    if fetched is not None and len(fetched):
      _save_year(safe, year, fetched)
      by_year[year] = fetched

  cur = _pull_year(client, symbol, y_now, y_now, avail_end)
  if cur is not None and len(cur):
    _save_year(safe, y_now, cur)
    by_year[y_now] = cur

  last_y = y_now - 1
  if last_y >= y0 and _need_refresh_last_year(safe, last_y, y_now):
    fetched = _pull_year(client, symbol, last_y, y_now, avail_end)
    if fetched is not None and len(fetched):
      _save_year(safe, last_y, fetched)
      by_year[last_y] = fetched

  if not by_year:
    raise RuntimeError(f'Databento returned no rows for {symbol} from {y0}')

  part = pd.concat([by_year[y] for y in sorted(by_year)], axis=0)
  part = part.sort_index()
  part = part[~part.index.duplicated(keep='last')]
  out = _ratio_adjust(part)
  return out.loc[out.index >= pd.Timestamp(f'{y0}-01-01')]

def getPriceHistoryFred(id, yrStart=SHARED_DICT['yrStart']):
  dtStart = f"{yrStart}-01-01"
  url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd={dtStart}"
  df = pd.read_csv(url, parse_dates=['observation_date'], index_col='observation_date')
  df.columns = [id]
  df = df.replace('.', np.nan).astype(float).dropna()
  df.index.name = None
  return df[id]

def getPriceHistoryNaver(und):
  code = re.match(r'^(\d{6})', str(und).strip()).group(1)
  now = pendulum.now('Asia/Seoul')
  url = (f"https://api.finance.naver.com/siseJson.naver"
         f"?symbol={code}&requestType=1&startTime={now.subtract(days=10).format('YYYYMMDD')}"
         f"&endTime={now.format('YYYYMMDD')}&timeframe=day")
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    r = curl_cffi.Session(impersonate="chrome").get(
      url, headers={'Referer': f'https://finance.naver.com/item/main.naver?code={code}'}, timeout=15)
  rows = re.findall(r'\[\s*"(\d{8})"\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)', r.text)
  df = pd.DataFrame(rows, columns=['date', 'Open', 'High', 'Low', 'Close'])
  df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
  df = df.set_index('date')
  for c in ul.spl('Open,High,Low,Close'):
    df[c] = df[c].astype(float)
  df['Volume'] = 0
  return df.sort_index()

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

def naverPatchHynix(df):
  now = pendulum.now('Asia/Seoul')
  if now.hour * 60 + now.minute < 15 * 60 + 30:
    return df
  today = now.date()
  if df is None or len(df) == 0 or df.index.max().date() >= today:
    return df
  try:
    nv = getPriceHistoryNaver('000660')
  except Exception:
    return df
  if len(nv) == 0 or nv.index.max().date() != today:
    return df
  df = pd.concat([df, nv.iloc[[-1]]])
  return df[~df.index.duplicated(keep='last')].sort_index()