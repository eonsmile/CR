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
  l = ul.spl('TAA,COS,HMR,IBS,JMR,RSS,GMR,WRS,SCI,VCA,VCA2,AHS,BTS,GEO,TNQ,WTS,ZBT')
  d = {
    'TAA': a,  # _TAA
    #####
    # Tier A MRs
    'COS': a,  # _MR
    'HMR': a,  # _HMR
    'IBS': a,  # _MR
    'JMR': a,  # _MR
    'RSS': a,  # _MR
    #####
    # Tier B MRs
    'GMR': b,  # _MR
    'WRS': b,  # _MR
    #####
    # Hedges
    'SCI': b,  # *** review monthly ***
    'VCA': a,  # _ETC
    'VCA2': a, # _ETC
    #####
    # Etc
    'AHS': b,  # *** review monthly ***
    'BTS': a,  # _BTS
    'GEO': b,  # *** review monthly ***
    'TNQ': b,  # TradeStation
    'WTS': b,  # _ETC
    'ZBT': a,  # *** review monthly ***
  }
  st.write(f"Total weights: {np.sum(list(d.values())):.2f}")
######
#    UPI: 50.06          UPI3y: 78.36          Sharpe: 5.49          Cagr: 42.8%          MaxDD: 4.1%
#    UPI: 50.74          UPI3y: 76.66          Sharpe: 4.54          Cagr: 42.8%          MaxDD: 3.9%
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
  runJMR(y)
  st.divider()
  runRSS(y)
  st.divider()
  #####
  # Tier B MRs
  runGMR(y)
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
  #####
  # Etc
  runAHS(y)
  st.divider()
  runBTS(y)
  st.divider()
  runGEO(y)
  st.divider()
  runTNQ(y)
  st.divider()
  runWTS(y)
  st.divider()
  runZBT(y)
  st.divider()

runAlpha(y, isSkipTitle=True)
