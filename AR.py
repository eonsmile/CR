import QuantLib as ql
import StratLib as sl
import UtilLib as ul
import streamlit as st
import numpy as np
import pandas as pd
import ctypes
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

########
# Params
########
# AGENT — Alpha book run log SoT is C:\OneDrive\Py3\AR.md (not comments here).
# When you run AR from an agent (preferred: Lab/adhoc/_scratch/run_ar_headless.py, or
# any headless session that mocks streamlit then calls runAlpha), runAlpha appends one
# aligned row to AR.md after the Alpha bt (UPI/UPI3y/Sharpe/CAGR/MaxDD/sumW/MTD/YTD).
# Pass note=… (or env AR_MD_NOTE) for the Note column. Streamlit UI runs do not stamp.
# Do not write UPI history into this file.
if ctypes.WinDLL('User32.dll').GetKeyState(0x14) & 1:
  print('')
  ul.cPrint('[CAPS]', 'yellow',isReverse=True)
  isRunSystems=False
else:
  isRunSystems = True

def _isHeadlessAr() -> bool:
  """True when streamlit is mocked (agent / run_ar_headless), not interactive UI."""
  st_mod = sys.modules.get('streamlit')
  return isinstance(st_mod, MagicMock)

###########
# Functions
###########
def runAlpha(yrStart, isSkipTitle=False, note=None, isStampArMd=None):
  script = 'Alpha'
  if not isSkipTitle:
    st.header(script)
  #####
  # Within each group, scripts are alphabetically sorted (l / weights / run* match StratLib).
  a=.2
  b=.08
  l = ul.spl('TAA,COS,COS2,CUT,IBS,IBS2,IGQ,MTQ,RSS,STK,YPS,COS3,HMR,JMR,SCI,VCA,VCA2,VCA3,VCA4,BCS,NGS,SIS,WTS,AHS,GEO,ZBT')

  d = {
    # Momentum
    'TAA': a,  # _TAA
    #####
    # US Broad MRs
    'COS': a,  # _MR
    'COS2': b,  # _MR
    'CUT': b,  # _MR
    'IBS': b,  # _MR
    'IBS2': b,  # _MR
    'IGQ': b,  # _MR    
    'MTQ': b,  # _MR
    'RSS': b,  # _MR
    'STK': b,  # _MR
    'YPS': b,  # _MR
    #####
    # Other MRs
    'COS3': b,  # _MR2
    'HMR': b,  # _HMR
    'JMR': b,  # _MR2
    #####
    # Hedges
    'SCI': b,  # *** review monthly ***
    'VCA': a,  # _HEDGES
    'VCA2': b,  # _HEDGES
    'VCA3': b,  # _HEDGES
    'VCA4': b,  # _HEDGES
    #####
    # Alt1
    'BCS': b,  # _BCS
    'NGS': b,  # _ALT1
    'SIS': b,  # _ALT1
    'WTS': b,  # _ALT1
    #####
    # Etc
    'AHS': b,  # *** review monthly ***
    'GEO': b,  # *** review monthly ***
    'ZBT': b,  # *** review monthly ***
  }
  sumW = float(np.sum(list(d.values())))
  st.write(f"Total weights: {sumW:.2f}")
######

  tickers = d.keys() - l
  if tickers:
    dp, dw, _, _ = ql.btSetup(tickers, yrStart=yrStart - 1)
  else:
    anchor = ul.cachePersist('r', l[0])
    if isinstance(anchor, pd.DataFrame):
      anchor = anchor.iloc[:, 0]
    dp = pd.DataFrame(index=anchor.index)
    dw = dp.copy()
    dw[:] = np.nan
  pe = ql.endpoints(dw)
  for und in l:
    dw[und]=np.nan
    dp[und]=np.nan
  for und in d.keys():
    dw.iloc[pe,dw.columns.get_loc(und)]=d[und]
  for und in l:
    dp[und] = ql.applyDates(ul.cachePersist('r', und), dp)
  #####
  dp = dp.bfill()
  ql.bt(script, dp, dw, yrStart)
  #####
  st.header('Corrs')
  ql.stWriteDf(dp.pct_change().corr().round(3))
  #####
  # Agent / headless: stamp AR.md (SoT). Interactive Streamlit skips.
  if isStampArMd is None:
    isStampArMd = _isHeadlessAr()
  if isStampArMd:
    note_s = (note if note is not None else os.environ.get('AR_MD_NOTE', 'headless AR')).strip()
    adhoc = Path(r'C:\OneDrive\Py3\Lab\adhoc')
    if str(adhoc) not in sys.path:
      sys.path.insert(0, str(adhoc))
    import afr_ar_stats as _ar_md
    row = _ar_md.stamp_alpha_to_ar_md(note=note_s or 'headless AR', sum_w=sumW)
    print(f'AR.md += {row}', flush=True)

######
# Main
######
z='Alpha Reporter'
st.set_page_config(page_title=z)
st.title(z)

chosenYear = 2017
st.write('')
y = int(chosenYear)
if isRunSystems:
  # Presentation order matches `l` / StratLib Core order.
  # Momentum
  sl.runTAA(y)
  st.divider()
  #####
  # US Broad MRs
  sl.runCOS(y)
  st.divider()
  sl.runCOS2(y)
  st.divider()
  sl.runCUT(y)
  st.divider()
  sl.runIBS(y)
  st.divider()
  sl.runIBS2(y)
  st.divider()
  sl.runIGQ(y)
  st.divider()
  sl.runMTQ(y)
  st.divider()
  sl.runRSS(y)
  st.divider()
  sl.runSTK(y)
  st.divider()
  sl.runYPS(y)
  st.divider()
  #####
  # Other MRs
  sl.runCOS3(y)
  st.divider()
  sl.runHMR(y)
  st.divider()
  sl.runJMR(y)
  st.divider()
  #####
  # Hedges
  sl.runSCI(y)
  st.divider()
  sl.runVCA(y)
  st.divider()
  sl.runVCA2(y)
  st.divider()
  sl.runVCA3(y)
  st.divider()
  sl.runVCA4(y)
  st.divider()
  #####
  # Alt1
  sl.runBCS(y)
  st.divider()
  sl.runNGS(y)
  st.divider()
  sl.runSIS(y)
  st.divider()
  sl.runWTS(y)
  st.divider()
  #####
  # Etc
  sl.runAHS(y)
  st.divider()
  sl.runGEO(y)
  st.divider()
  sl.runZBT(y)
  st.divider()

runAlpha(y, isSkipTitle=True)
