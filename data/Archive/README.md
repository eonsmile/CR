```
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
                   'COM,DBMF,JEGA.LSE,PFMN.TO,'
                   'AHLT,ASMF,CTA,HFMF,ISMF,KMLM,TFPN,'
                   'CAOS,GRIN,HARD,HECA,IFLO,HFGM,HGER,QALT,VFLO,'                   
                   'ENCO.LSE,DFNS.LSE,GCOW,IALT,ICOW,ORR,PFIX,RARE.LSE,TAIL,WCOA.LSE,'
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
    elif und == 'JEGA.LSE':
      dtStart = '2023-12-29'
    elif und == 'PFMN.TO':
      dtStart = '2019-7-31'
    #####
    elif und=='AHLT':
      dtStart = '2023-8-31'
    elif und=='ASMF':
      dtStart = '2024-5-31'
    elif und=='CTA':
      dtStart = '2022-3-31'
    elif und=='HFMF':
      dtStart = '2025-7-31'
    elif und=='ISMF':
      dtStart = '2025-3-31'
    # KMLM
    elif und=='TFPN':
      dtStart = '2023-7-31'
    #####
    elif und == 'CAOS':
      dtStart = '2023-3-31'
    elif und == 'GRIN':
      dtStart = '2025-6-30'
    elif und=='HARD':
      dtStart = '2023-3-31'
    elif und == 'HECA':
      dtStart = '2025-7-31'
    elif und == 'IFLO':
      dtStart = '2025-6-30'
    elif und=='HFGM':
      dtStart='2025-4-30'
    elif und=='HGER':
      dtStart = '2022-2-28'
    elif und == 'QALT':
      dtStart = '2025-8-29'
    elif und == 'VFLO':
      dtStart = '2023-6-30'
    #####
    elif und=='ENCO.LSE':
      dtStart='2021-8-31'
    elif und=='DFNS.LSE':
      dtStart='2023-4-28'
    elif und=='GCOW':
      dtStart='2016-2-29'
    elif und=='IALT':
      dtStart='2025-12-31'
    elif und=='ICOW':
      dtStart='2017-6-30'
    elif und=='ORR':
      dtStart = '2025-1-31'
    elif und=='PFIX':
      dtStart='2021-5-28'
    elif und=='RARE.LSE':
      dtStart='2024-4-30'
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
```