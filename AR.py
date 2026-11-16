from QuantLib import *
import UtilLib as ul
import ctypes

########
# Params
########
if ctypes.WinDLL('User32.dll').GetKeyState(0x14) & 1:
  print('')
  ul.cPrint('[CAPS]', 'yellow',isReverse=True)
  isRunSystems=False
else:
  isRunSystems = True

###########
# Functions
###########
def runAlpha(yrStart, isSkipTitle=False):
  script = 'Alpha'
  if not isSkipTitle:
    st.header(script)
  #####
  # Within each group, scripts are alphabetically sorted (l / weights / run* match QuantLib).
  a=.2
  b=.08
  l = ul.spl('TAA,COS,IBS,RSS,BBS,GMR,HMR,IBS2,JMR,MTM,STK,YPS,SCI,VCA,VCA2,VCA3,BCS,NGS,SIS,WTS,AHS,GEO,ZBT')
  d = {
    'TAA': a,  # _TAA
    #####
    # Tier A MRs
    'COS': a,  # _MR
    'IBS': a,  # _MR
    'RSS': a,  # _MR
    #####
    # Tier B MRs
    'BBS': b,  # _MR
    'GMR': b,  # _MR
    'HMR': b,  # _HMR
    'IBS2': b,  # _MR
    'JMR': b,  # _MR
    'MTM': b,  # _MR
    'STK': b,  # _MR
    'YPS': b,  # _MR
    #####
    # Hedges
    'SCI': b,  # *** review monthly ***
    'VCA': a,  # _HEDGES
    'VCA2': b,  # _HEDGES
    'VCA3': b,  # _HEDGES
    #####
    # Alt1
    'BCS': a,  # _BCS
    'NGS': b,  # _ALT1
    'SIS': b,  # _ALT1; spike∨ChinaLongHoliday VT0.32 nPost=1 (2026-11-16)
    'WTS': b,  # _ALT1
    #####
    # Etc
    'AHS': b,  # *** review monthly ***
    'GEO': b,  # *** review monthly ***
    'ZBT': b,  # *** review monthly ***
  }
  st.write(f"Total weights: {np.sum(list(d.values())):.2f}")
######
#  UPI: 54.21          UPI3y: 81.88          Sharpe: 4.67          Cagr: 46.2%          MaxDD: 4.4% (2026-11-15 06:15; +NGS @b VT0.32; BCS/SIS rename; Alt1; sumW 2.56)
#  UPI: 55.13          UPI3y: 81.88          Sharpe: 4.70          Cagr: 46.5%          MaxDD: 4.4% (2026-11-15 16:25; NGS OR @b VT0.24 pre-EIA; Alt1; sumW 2.56)
#  UPI: 55.58          UPI3y: 82.67          Sharpe: 4.72          Cagr: 46.7%          MaxDD: 4.4% (2026-11-15 18:54; WNGSR calendar rebuild; NGS OR @b VT0.24; sumW 2.56)

#####
  tickers = d.keys() - l
  if tickers:
    dp, dw, _, _ = btSetup(tickers, yrStart=yrStart - 1)
  else:
    anchor = ul.cachePersist('r', l[0])
    if isinstance(anchor, pd.DataFrame):
      anchor = anchor.iloc[:, 0]
    dp = pd.DataFrame(index=anchor.index)
    dw = dp.copy()
    dw[:] = np.nan
  pe = endpoints(dw)
  for und in l:
    dw[und]=np.nan
    dp[und]=np.nan
  for und in d.keys():
    dw.iloc[pe,dw.columns.get_loc(und)]=d[und]
  for und in l:
    dp[und] = applyDates(ul.cachePersist('r', und), dp)
  #####
  dp = dp.bfill()
  bt(script, dp, dw, yrStart)
  #####
  st.header('Corrs')
  stWriteDf(dp.pct_change().corr().round(3))

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
  # Presentation order matches `l` / QuantLib Core order.
  runTAA(y)
  st.divider()
  #####
  # Tier A MRs
  runCOS(y)
  st.divider()
  runIBS(y)
  st.divider()
  runRSS(y)
  st.divider()
  #####
  # Tier B MRs
  runBBS(y)
  st.divider()
  runGMR(y)
  st.divider()
  runHMR(y)
  st.divider()
  runIBS2(y)
  st.divider()
  runJMR(y)
  st.divider()
  runMTM(y)
  st.divider()
  runSTK(y)
  st.divider()
  runYPS(y)
  st.divider()
  #####
  # Hedges
  runSCI(y)
  st.divider()
  runVCA(y)
  st.divider()
  runVCA2(y)
  st.divider()
  runVCA3(y)
  st.divider()
  #####
  # Alt1
  runBCS(y)
  st.divider()
  runNGS(y)
  st.divider()
  runSIS(y)
  st.divider()
  runWTS(y)
  st.divider()
  #####
  # Etc
  runAHS(y)
  st.divider()
  runGEO(y)
  st.divider()
  runZBT(y)
  st.divider()

runAlpha(y, isSkipTitle=True)
