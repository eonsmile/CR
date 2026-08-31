###############
# Databento Library
###############
from UtilLib import SHARED_DICT
import numpy as np
import pandas as pd
import databento as db
import os
import re
import time

_DB_DATASET = 'GLBX.MDP3'
_DB_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'databento')
_DB_LAST_YEAR_DEADLINE_MD = (1, 15)
_DB_TZ_CT = 'America/Chicago'


def _db_api_key():
  key = os.getenv('DB_API_KEY', '').strip()
  if not key:
    raise RuntimeError('DB_API_KEY missing from .env')
  return key


def _db_normalize_symbol(und):
  s = str(und).strip()
  if re.fullmatch(r'[A-Za-z0-9]+\.[cnv]\.\d+', s):
    return s
  if re.fullmatch(r'[A-Za-z0-9]+', s):
    return f'{s}.v.0'
  return s


def _db_year_path(safe, year):
  return os.path.join(_DB_CACHE_DIR, f'{safe}_{year}.parquet')


def _db_parse_avail_end(val):
  return pd.Timestamp(str(val).replace('Z', '').split('+')[0])


def _db_dataset_range(client, schema, which):
  """metadata.get_dataset_range is free (no timeseries bill). which is 'start' or 'end'."""
  rng = client.metadata.get_dataset_range(dataset=_DB_DATASET)
  schemas = rng.get('schema') or {}
  avail = (schemas.get(schema) or {}).get(which) or rng.get(which)
  return _db_parse_avail_end(avail)


def _db_available_end(client, schema):
  return _db_dataset_range(client, schema, 'end')


def _db_available_start(client, schema):
  return _db_dataset_range(client, schema, 'start')


def _db_to_utc_ts(ts):
  t = pd.Timestamp(ts)
  if t.tzinfo is None:
    return t.tz_localize('UTC')
  return t.tz_convert('UTC')


def _db_utc_next_month(ts):
  t = _db_to_utc_ts(ts)
  if t.month == 12:
    return pd.Timestamp(year=t.year + 1, month=1, day=1, tz='UTC')
  return pd.Timestamp(year=t.year, month=t.month + 1, day=1, tz='UTC')


def _db_fmt_utc(ts):
  return _db_to_utc_ts(ts).strftime('%Y-%m-%dT%H:%M:%S')


def _db_pull_to_df(client, symbol, schema, start, end, retries=4):
  """Historical timeseries.get_range — this is what Databento bills."""
  req_start, req_end = start, end
  last_err = None
  for attempt in range(retries):
    try:
      data = client.timeseries.get_range(
        dataset=_DB_DATASET,
        schema=schema,
        symbols=symbol,
        stype_in='continuous',
        start=req_start,
        end=req_end,
      )
      df = data.to_df()
      if df is None or len(df) == 0:
        return None
      return df
    except Exception as e:
      last_err = e
      msg = str(e)
      m_end = re.search(r"available_end['\":\s]+['\"]?([0-9T:\.\-\+Z]+)", msg)
      if not m_end:
        m_end = re.search(r"end time before ([0-9T:\.\+\-Z]+)", msg)
      if m_end and attempt < retries - 1:
        req_end = m_end.group(1).rstrip('Z').split('+')[0]
        continue
      m_start = re.search(r"start time after ([0-9T:\.\+\-Z]+)", msg)
      if not m_start:
        m_start = re.search(r"available_start['\":\s]+['\"]?([0-9T:\.\+\-Z]+)", msg)
      if m_start and attempt < retries - 1:
        req_start = m_start.group(1).rstrip('Z').split('+')[0]
        continue
      if 'data_start_before_available' in msg or 'dataset_unavailable_range' in msg:
        return None
      time.sleep(1.2 * (attempt + 1))
  if last_err is not None:
    raise last_err
  return None


def _db_normalize_raw_daily(df):
  if df is None or len(df) == 0:
    return None
  part = df.reset_index() if 'ts_event' not in getattr(df, 'columns', []) else df.copy()
  if 'ts_event' not in part.columns:
    part = part.reset_index()
  part['date'] = pd.to_datetime(part['ts_event']).dt.tz_convert(None).dt.normalize()
  part = part.sort_values('date').drop_duplicates('date', keep='last').set_index('date')
  out = {
    'Open': part['open'].astype(float),
    'High': part['high'].astype(float),
    'Low': part['low'].astype(float),
    'Close': part['close'].astype(float),
    'Volume': part['volume'].astype(float),
  }
  if 'instrument_id' in part.columns:
    out['instrument_id'] = part['instrument_id'].astype(int)
  return pd.DataFrame(out)


def _db_normalize_raw_1m(df):
  if df is None or len(df) == 0:
    return None
  part = df.reset_index() if 'ts_event' not in getattr(df, 'columns', []) else df.copy()
  if 'ts_event' not in part.columns:
    part = part.reset_index()
  ts = pd.to_datetime(part['ts_event'], utc=True)
  cols = {
    'open': part['open'].to_numpy(dtype=float),
    'high': part['high'].to_numpy(dtype=float),
    'low': part['low'].to_numpy(dtype=float),
    'close': part['close'].to_numpy(dtype=float),
    'volume': part['volume'].to_numpy(dtype=float),
  }
  if 'instrument_id' not in part.columns:
    raise RuntimeError('ohlcv-1m missing instrument_id')
  cols['instrument_id'] = part['instrument_id'].to_numpy(dtype=np.int64)
  out = pd.DataFrame(cols, index=pd.DatetimeIndex(ts, name='ts_event')).sort_index()
  out = out[~out.index.duplicated(keep='last')]
  out = out.dropna(subset=['open', 'high', 'low', 'close'])
  _db_require_1m_instrument_id(out, 'ohlcv-1m')
  return out


def _db_load_year(safe, year):
  path = _db_year_path(safe, year)
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


def _db_save_year(safe, year, df):
  if df is None or len(df) == 0:
    return
  os.makedirs(_DB_CACHE_DIR, exist_ok=True)
  df.to_parquet(_db_year_path(safe, year))


def _db_need_refresh_last_year(safe, last_y, y_now):
  path = _db_year_path(safe, last_y)
  if not os.path.exists(path) or os.path.getsize(path) == 0:
    return True
  mtime = pd.Timestamp(os.path.getmtime(path), unit='s')
  deadline = pd.Timestamp(year=y_now, month=_DB_LAST_YEAR_DEADLINE_MD[0], day=_DB_LAST_YEAR_DEADLINE_MD[1])
  return mtime < deadline


def _db_require_1m_instrument_id(df, where):
  """1m cache is required to carry instrument_id on every row (id-less history was retired)."""
  if df is None or len(df) == 0:
    return
  if 'instrument_id' not in df.columns:
    raise RuntimeError(f'{where}: 1m missing instrument_id')
  n_nan = int(df['instrument_id'].isna().sum())
  if n_nan:
    raise RuntimeError(f'{where}: instrument_id has {n_nan} nulls')


def _db_ratio_adjust(part, daily=True):
  """Ratio-adjust roll gaps on instrument_id changes. Scale-invariant for ATR/PO away from rolls."""
  close_raw = part['Close'].astype(float).to_numpy()
  adj = {
    'Open': part['Open'].astype(float).to_numpy().copy(),
    'High': part['High'].astype(float).to_numpy().copy(),
    'Low': part['Low'].astype(float).to_numpy().copy(),
    'Close': close_raw.copy(),
  }
  if not daily:
    _db_require_1m_instrument_id(part, '_db_ratio_adjust')
  if 'instrument_id' in part.columns and part['instrument_id'].notna().any():
    ids = part['instrument_id'].astype(float).to_numpy()
    chg = np.where(ids[1:] != ids[:-1])[0] + 1
    for i in chg[::-1]:
      if np.isnan(ids[i]) or np.isnan(ids[i - 1]):
        continue
      prev_px, cur_px = close_raw[i - 1], close_raw[i]
      if prev_px == 0 or np.isnan(prev_px) or np.isnan(cur_px):
        continue
      factor = cur_px / prev_px
      for a in adj.values():
        a[:i] *= factor
  vol = part['Volume'].fillna(0)
  out = pd.DataFrame({
    'Open': adj['Open'],
    'High': adj['High'],
    'Low': adj['Low'],
    'Close': adj['Close'],
    'Volume': vol.round(0).astype(np.int64).to_numpy() if daily else vol.to_numpy(dtype=float),
  }, index=part.index)
  if daily:
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None).normalize()
    out.index.name = 'date'
  else:
    out.index.name = 'ts_event'
  return out.sort_index().round(10)


def _db_intraday_dir(symbol):
  root = symbol.split('.')[0].upper()
  rest = symbol[len(root):]
  if rest.lower() in ('', '.v.0'):
    name = f'{root}_intraday'
  else:
    name = f"{symbol.replace('.', '_')}_intraday"
  return os.path.join(_DB_CACHE_DIR, name)


def _db_intraday_month_path(dest_dir, symbol, year, month):
  return os.path.join(dest_dir, f'{symbol}_1m_{int(year):04d}{int(month):02d}.parquet')


def _db_intraday_list_stems(dest_dir, symbol):
  if not os.path.isdir(dest_dir):
    return []
  prefix = f'{symbol}_1m_'
  stems = []
  for fn in os.listdir(dest_dir):
    if not fn.startswith(prefix) or not fn.endswith('.parquet'):
      continue
    stem = fn[len(prefix):-8]
    if len(stem) == 6 and stem.isdigit() and os.path.getsize(os.path.join(dest_dir, fn)) >= 100:
      stems.append(stem)
  return sorted(stems)


def _db_intraday_write_months(dest_dir, symbol, df):
  if df is None or len(df) == 0:
    return
  x = df.copy()
  _db_require_1m_instrument_id(x, f'{symbol} 1m write')
  if x.index.tz is None:
    x.index = x.index.tz_localize('UTC')
  else:
    x.index = x.index.tz_convert('UTC')
  x.index.name = 'ts_event'
  os.makedirs(dest_dir, exist_ok=True)
  for (y, m), g in x.groupby([x.index.year, x.index.month]):
    path = _db_intraday_month_path(dest_dir, symbol, y, m)
    if os.path.exists(path) and os.path.getsize(path) > 0:
      old = pd.read_parquet(path)
      if old.index.tz is None:
        old.index = old.index.tz_localize('UTC')
      else:
        old.index = old.index.tz_convert('UTC')
      g = pd.concat([old, g]).sort_index()
    g = g[~g.index.duplicated(keep='last')]
    _db_require_1m_instrument_id(g, path)
    g.to_parquet(path)


def _db_intraday_load_months(dest_dir, symbol):
  if not os.path.isdir(dest_dir):
    return None
  parts = []
  prefix = f'{symbol}_1m_'
  for fn in sorted(os.listdir(dest_dir)):
    if not fn.startswith(prefix) or not fn.endswith('.parquet'):
      continue
    stem = fn[len(prefix):-8]
    if not (len(stem) == 6 and stem.isdigit()):
      continue
    path = os.path.join(dest_dir, fn)
    if os.path.getsize(path) < 100:
      continue
    try:
      df = pd.read_parquet(path)
    except Exception:
      continue
    if df is None or len(df) == 0:
      continue
    _db_require_1m_instrument_id(df, path)
    if df.index.tz is None:
      df.index = df.index.tz_localize('UTC')
    else:
      df.index = df.index.tz_convert('UTC')
    df.index.name = 'ts_event'
    parts.append(df)
  if not parts:
    return None
  out = pd.concat(parts).sort_index()
  out = out[~out.index.duplicated(keep='last')]
  _db_require_1m_instrument_id(out, dest_dir)
  return out


def _db_intraday_remold_legacy(symbol):
  """Copy existing 1m parquets into data/databento/{root}_intraday/ monthly files. No API."""
  legacy = os.path.join(_DB_CACHE_DIR, 'intraday')
  dest = _db_intraday_dir(symbol)
  if not os.path.isdir(legacy):
    return
  prefix = f'{symbol}_1m_'
  if os.path.isdir(dest):
    have = [
      fn for fn in os.listdir(dest)
      if fn.startswith(prefix) and fn.endswith('.parquet')
      and len(fn[len(prefix):-8]) == 6 and fn[len(prefix):-8].isdigit()
    ]
    if have:
      return
  chunks = []
  for fn in os.listdir(legacy):
    if not fn.startswith(prefix) or not fn.endswith('.parquet'):
      continue
    path = os.path.join(legacy, fn)
    if os.path.getsize(path) < 100:
      continue
    try:
      df = pd.read_parquet(path)
    except Exception:
      continue
    if df is None or len(df) == 0 or 'open' not in df.columns:
      continue
    chunks.append(df)
  if not chunks:
    return
  raw = pd.concat(chunks).sort_index()
  if raw.index.tz is None:
    raw.index = raw.index.tz_localize('UTC')
  else:
    raw.index = raw.index.tz_convert('UTC')
  raw = raw[~raw.index.duplicated(keep='last')]
  _db_require_1m_instrument_id(raw, f'legacy remold {symbol}')
  keep = [c for c in ('open', 'high', 'low', 'close', 'volume', 'instrument_id') if c in raw.columns]
  _db_intraday_write_months(dest, symbol, raw[keep])


def _db_resample_eth(df, minutes):
  x = df.copy()
  if x.index.tz is None:
    x.index = x.index.tz_localize('UTC')
  x = x.tz_convert(_DB_TZ_CT)
  t = x.index.hour * 100 + x.index.minute
  x = x[(t < 1600) | (t >= 1700)]
  ohlc = x.resample(f'{int(minutes)}min', label='left', closed='left').agg(
    Open=('Open', 'first'),
    High=('High', 'max'),
    Low=('Low', 'min'),
    Close=('Close', 'last'),
    Volume=('Volume', 'sum'),
  ).dropna(subset=['Open', 'High', 'Low', 'Close'])
  tt = ohlc.index.hour * 100 + ohlc.index.minute
  ohlc = ohlc[(tt < 1600) | (tt >= 1700)]
  return ohlc[ohlc['Volume'] > 0]


def getPriceHistoryDB(und, yrStart=SHARED_DICT['yrStart']):
  """Databento continuous futures OHLCV — same shape as getPriceHistory (EODHD).

  Root 'HG' → HG.v.0. Yearly parquet under data/databento/:
    - prior years: reuse cache if present, else pull once (earlier yrStart backfills missing years)
    - current year: always live-pull (overwrite cache) — days/months off still catch up
    - last year: re-pull unless file mtime is on/after Jan 15 of the current year
  Then ratio-adjust roll gaps. Auth: DB_API_KEY in .env.
  """
  symbol = _db_normalize_symbol(und)
  safe = symbol.replace('.', '_')
  y0 = int(yrStart)

  client = db.Historical(_db_api_key())
  try:
    avail_end = _db_available_end(client, 'ohlcv-1d')
  except Exception:
    avail_end = pd.Timestamp.utcnow().normalize()
  try:
    avail_start = _db_available_start(client, 'ohlcv-1d')
  except Exception:
    avail_start = None
  y_now = int(avail_end.year)
  y_first = y0 if avail_start is None else max(y0, int(pd.Timestamp(avail_start).year))

  by_year = {}
  for year in range(y_first, y_now):
    cached = _db_load_year(safe, year)
    if cached is not None:
      by_year[year] = cached
      continue
    start_y = f'{year}-01-01'
    if avail_start is not None and year == int(pd.Timestamp(avail_start).year):
      start_y = pd.Timestamp(avail_start).strftime('%Y-%m-%d')
    raw = _db_pull_to_df(client, symbol, 'ohlcv-1d', start_y, f'{year + 1}-01-01')
    fetched = _db_normalize_raw_daily(raw)
    if fetched is not None and len(fetched):
      fetched = fetched.loc[fetched.index.year == year]
    if fetched is not None and len(fetched):
      _db_save_year(safe, year, fetched)
      by_year[year] = fetched

  start_y = f'{y_now}-01-01'
  raw = _db_pull_to_df(client, symbol, 'ohlcv-1d', start_y, avail_end.strftime('%Y-%m-%dT%H:%M:%S'))
  cur = _db_normalize_raw_daily(raw)
  if cur is not None and len(cur):
    cur = cur.loc[cur.index.year == y_now]
  if cur is not None and len(cur):
    _db_save_year(safe, y_now, cur)
    by_year[y_now] = cur
  else:
    cached_cur = _db_load_year(safe, y_now)
    if cached_cur is not None:
      by_year[y_now] = cached_cur

  last_y = y_now - 1
  if last_y >= y_first and _db_need_refresh_last_year(safe, last_y, y_now):
    raw = _db_pull_to_df(client, symbol, 'ohlcv-1d', f'{last_y}-01-01', f'{last_y + 1}-01-01')
    fetched = _db_normalize_raw_daily(raw)
    if fetched is not None and len(fetched):
      fetched = fetched.loc[fetched.index.year == last_y]
    if fetched is not None and len(fetched):
      _db_save_year(safe, last_y, fetched)
      by_year[last_y] = fetched

  if not by_year:
    raise RuntimeError(f'Databento returned no rows for {symbol} from {y0}')

  part = pd.concat([by_year[y] for y in sorted(by_year)], axis=0)
  part = part.sort_index()
  part = part[~part.index.duplicated(keep='last')]
  out = _db_ratio_adjust(part, daily=True)
  out = out.loc[out.index >= pd.Timestamp(f'{y0}-01-01')]
  if len(out) == 0:
    raise RuntimeError(f'Databento daily empty for {symbol} from {y0}')
  if int(out.index.min().year) > y_first + 1:
    raise RuntimeError(
      f'{symbol} daily backfill incomplete: data starts {out.index.min().date()}, wanted {y_first}')
  last_d = pd.Timestamp(out.index.max()).normalize()
  end_d = pd.Timestamp(avail_end)
  if end_d.tzinfo is not None:
    end_d = end_d.tz_convert('UTC').tz_localize(None)
  end_d = end_d.normalize()
  if last_d < end_d - pd.Timedelta(days=10):
    raise RuntimeError(
      f'{symbol} daily catch-up incomplete: last={last_d.date()} available_end={end_d.date()}')
  return out


def _db_intraday_fetch_write(client, symbol, dest, start, end):
  """Billed ohlcv-1m in calendar-month chunks → monthly parquet. No-op on empty chunks."""
  cur = _db_to_utc_ts(start)
  end_ts = _db_to_utc_ts(end)
  while cur < end_ts:
    chunk_end = min(_db_utc_next_month(cur), end_ts)
    raw = _db_pull_to_df(client, symbol, 'ohlcv-1m', _db_fmt_utc(cur), _db_fmt_utc(chunk_end))
    part = _db_normalize_raw_1m(raw)
    if part is not None and len(part):
      _db_intraday_write_months(dest, symbol, part)
    cur = chunk_end


def _db_intraday_fill_missing_months(client, symbol, dest):
  """Pull calendar months missing between the first and last cached files."""
  stems = _db_intraday_list_stems(dest, symbol)
  if len(stems) < 2:
    return
  have = set(stems)
  y, m = int(stems[0][:4]), int(stems[0][4:6])
  y1, m1 = int(stems[-1][:4]), int(stems[-1][4:6])
  while (y, m) < (y1, m1):
    m += 1
    if m == 13:
      m, y = 1, y + 1
    if (y, m) >= (y1, m1):
      break
    if f'{y:04d}{m:02d}' in have:
      continue
    start = pd.Timestamp(year=y, month=m, day=1, tz='UTC')
    _db_intraday_fetch_write(client, symbol, dest, start, _db_utc_next_month(start))


def getPriceHistoryDBIntraday(und, yrStart=SHARED_DICT['yrStart'], intervalMins=1, live=True):
  """GLBX continuous 1m cache, ratio-adjust rolls, optional 5m/10m ETH resample.

  Store raw 1m monthly parquet under data/databento/{ROOT}_intraday/ (e.g. NQ_intraday).
  live=True: metadata.get_dataset_range is free. timeseries.get_range (ohlcv-1m) only for
  gaps, in calendar-month chunks: (1) backfill yrStart → first cached ts if the tape starts
  after yrStart; (2) missing months between first and last file; (3) catch-up last ts + 1m
  → available_end. Already-cached closed months are not re-billed.
  live=False never calls Databento.

  intervalMins: 1 (UTC 1m, halt bars kept), 5 or 10 (ETH resample, 16:00-17:00 CT dropped).
  """
  intervalMins = int(intervalMins)
  if intervalMins not in (1, 5, 10):
    raise ValueError('intervalMins must be 1, 5, or 10')
  symbol = _db_normalize_symbol(und)
  dest = _db_intraday_dir(symbol)
  _db_intraday_remold_legacy(symbol)
  y0 = pd.Timestamp(f'{int(yrStart)}-01-01', tz='UTC')

  start0 = avail_end = None
  if live:
    client = db.Historical(_db_api_key())
    try:
      avail_end = _db_to_utc_ts(_db_available_end(client, 'ohlcv-1m'))
    except Exception:
      avail_end = _db_to_utc_ts(pd.Timestamp.utcnow())
    try:
      avail_start = _db_to_utc_ts(_db_available_start(client, 'ohlcv-1m'))
    except Exception:
      avail_start = None
    start0 = y0 if avail_start is None or y0 >= avail_start else avail_start
    cached = _db_intraday_load_months(dest, symbol)
    first_ts = last_ts = None
    if cached is not None and len(cached):
      first_ts = _db_to_utc_ts(cached.index.min())
      last_ts = _db_to_utc_ts(cached.index.max())
    if last_ts is None:
      _db_intraday_fetch_write(client, symbol, dest, start0, avail_end)
    else:
      if first_ts > start0:
        _db_intraday_fetch_write(client, symbol, dest, start0, first_ts)
      _db_intraday_fill_missing_months(client, symbol, dest)
      if last_ts + pd.Timedelta(minutes=1) < avail_end:
        _db_intraday_fetch_write(
          client, symbol, dest, last_ts + pd.Timedelta(minutes=1), avail_end)

  raw = _db_intraday_load_months(dest, symbol)
  if raw is None or len(raw) == 0:
    raise RuntimeError(f'no intraday cache for {symbol} under {dest}')

  if live and start0 is not None and avail_end is not None:
    first_ts = _db_to_utc_ts(raw.index.min())
    last_ts = _db_to_utc_ts(raw.index.max())
    if first_ts > start0 + pd.Timedelta(days=10):
      raise RuntimeError(
        f'{symbol} 1m backfill incomplete: first={first_ts} wanted>={start0}')
    if last_ts + pd.Timedelta(days=10) < avail_end:
      raise RuntimeError(
        f'{symbol} 1m catch-up incomplete: last={last_ts} available_end={avail_end}')

  raw = raw.loc[raw.index >= y0]
  titled = pd.DataFrame({
    'Open': raw['open'].astype(float),
    'High': raw['high'].astype(float),
    'Low': raw['low'].astype(float),
    'Close': raw['close'].astype(float),
    'Volume': raw['volume'].astype(float),
  }, index=raw.index)
  titled['instrument_id'] = raw['instrument_id']
  adj = _db_ratio_adjust(titled, daily=False)
  if intervalMins == 1:
    return adj
  return _db_resample_eth(adj, intervalMins)

