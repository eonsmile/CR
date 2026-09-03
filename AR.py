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
  a=.15
  b=.03
  l = ul.spl('TPP,TPP2,IBS,RSS,COS,DAX,JMR,SCI,VCA,BTS,COM,GEO,SSS,HNX,MMQ')
  d = {
    # Systems
    'TPP': a,  # *** review monthly ***
    'TPP2': a,  # _TPP2
    #####
    'IBS': a, # _MR
    'RSS': a, # _MR
    'COS': a,  # _MR
    'DAX': a,  # _MR
    'JMR': a,  # _MR
    #####
    'SCI': a,  # *** review monthly ***
    'VCA': a, # _ETC
    #####
    'BTS': a,  # _BTS
    'COM': a,  # _ETC
    'GEO': a,  # *** review monthly ***
    'SSS': a,  # _ETC
    #####
    'HNX': b, # _HNX
    'MMQ': b, # MMQ-Trader
  }
  st.write(f"Total weights: {np.sum(list(d.values())):.2f}")
######
#    Calmar: 10.62          MAR: 8.55          Sharpe: 4.24          Cagr: 36.4%          MaxDD: 4.3% # monthend Aug post BTS fix
#    Calmar: 10.95          MAR: 8.92          Sharpe: 4.27          Cagr: 37.5%          MaxDD: 4.2% # DAX
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

chosenYear = 2016
st.write('')
y = int(chosenYear)
if isRunSystems:
  runTPP(y)
  st.divider()
  runTPP2(y)
  st.divider()
  #####
  runIBS(y)
  st.divider()
  runRSS(y)
  st.divider()
  runCOS(y)
  st.divider()
  runDAX(y)
  st.divider()
  runJMR(y)
  st.divider()
  #####
  runSCI(y)
  st.divider()
  runVCA(y)
  st.divider()
  #####
  runBTS(y)
  st.divider()
  runCOM(y)
  st.divider()
  runGEO(y)
  st.divider()
  runSSS(y)
  st.divider()
  #####
  runHNX(y)
  st.divider()
  #runMMQ(y)
  #st.divider()

runAlpha(y, isSkipTitle=True)

