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
  b=.07
  l = ul.spl('TAA,IBS,RSS,COS,GMR,HMR,JMR,SCI,VCA,VCA2,AHS,BTS,GEO,HNX,WTS,ZBT')
  d = {
    # Systems
    'TAA': a,  # _TAA
    #####
    'IBS': a, # _MR
    'RSS': a, # _MR
    'COS': a,  # _MR
    'GMR': b,  # _MR
    'HMR': b,  # _HMR
    'JMR': a,  # _MR
    #####
    'SCI': a,  # *** review monthly ***
    'VCA': a, # _ETC
    'VCA2': a, # _ETC
    #####
    'AHS': b,  # *** review monthly ***
    'BTS': a,  # _BTS
    'GEO': b,  # *** review monthly ***
    'HNX': b, # _HNX
    'WTS': b,  # _ETC
    'ZBT': a,  # *** review monthly ***
  }
  st.write(f"Total weights: {np.sum(list(d.values())):.2f}")
######
#  UPI: 47.15          UPI3y: 81.20          Sharpe: 4.66          Cagr: 42.5%          MaxDD: 4.4%
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
  runTAA(y)
  st.divider()
  #####
  runIBS(y)
  st.divider()
  runRSS(y)
  st.divider()
  runCOS(y)
  st.divider()
  runGMR(y)
  st.divider()
  runHMR(y)
  st.divider()
  runJMR(y)
  st.divider()
  #####
  runSCI(y)
  st.divider()
  runVCA(y)
  st.divider()
  runVCA2(y)
  st.divider()
  #####
  runAHS(y)
  st.divider()
  runBTS(y)
  st.divider()
  runGEO(y)
  st.divider()
  runHNX(y)
  st.divider()
  runWTS(y)
  st.divider()
  runZBT(y)
  st.divider()

runAlpha(y, isSkipTitle=True)

