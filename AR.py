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
  a=.2
  b=.08
  l = ul.spl('TAA,COS,HMR,IBS,RSS,BBS,GMR,IBS2,JMR,STK,WRS,SCI,VCA,VCA2,VCA3,AHS,BTS,GEO,WTS,ZBT')
  d = {
    'TAA': a,  # _TAA
    #####
    # Tier A MRs
    'COS': a,  # _MR
    'HMR': a,  # _HMR
    'IBS': a,  # _MR
    'RSS': a,  # _MR
    #####
    # Tier B MRs
    'BBS': b,  # _MR
    'GMR': b,  # _MR
    'IBS2': b,  # _MR — IWM; exit 0.8|hib TS6; VT 0.255 (ql.bt MDD<15%)
    'JMR': b,  # _MR — settled @b 2026-11-04 (LOO soft for a; ART R weak)
    'STK': b,  # _MR
    'WRS': b,  # _MR
    #####
    # Hedges
    'SCI': b,  # *** review monthly ***
    'VCA': a,  # _ETC
    'VCA2': b, # _ETC
    'VCA3': b, # _ETC
    #####
    # Etc
    'AHS': b,  # *** review monthly ***
    'BTS': a,  # _BTS
    'GEO': b,  # *** review monthly ***
    'WTS': b,  # _ETC
    'ZBT': a,  # *** review monthly ***
  }
  st.write(f"Total weights: {np.sum(list(d.values())):.2f}")
######
#  UPI: 49.19          UPI3y: 68.11          Sharpe: 4.45          Cagr: 43.2%          MaxDD: 4.1% (JMR a→b)
#  UPI: 50.46          UPI3y: 70.92          Sharpe: 4.46          Cagr: 44.5%          MaxDD: 4.5% (IBS2 @b)
#  UPI: 50.65          UPI3y: 69.99          Sharpe: 4.47          Cagr: 45.6%          MaxDD: 4.8% (b=0.08)
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
  dp = dp.bfill() # can try to see whether works or not
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
  # TAA (juggernaut)
  runTAA(y)
  st.divider()
  #####
  # Tier A MRs
  runCOS(y)
  st.divider()
  runHMR(y)
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
  runIBS2(y)
  st.divider()
  runJMR(y)
  st.divider()
  runSTK(y)
  st.divider()
  runWRS(y)
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
  # Etc
  runAHS(y)
  st.divider()
  runBTS(y)
  st.divider()
  runGEO(y)
  st.divider()
  runWTS(y)
  st.divider()
  runZBT(y)
  st.divider()

runAlpha(y, isSkipTitle=True)
