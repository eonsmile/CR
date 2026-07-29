from QuantLib import *
import UtilLib as ul
import ctypes

########
# Params
########
IS_CORRS=True
if ctypes.WinDLL('User32.dll').GetKeyState(0x14) & 1:
  print('')
  ul.cPrint('[CAPS]', 'yellow',isReverse=True)
  isRunSystems=True
else:
  isRunSystems = False

###########
# Functions
###########
def runAlpha(yrStart, isSkipTitle=False):
  script = 'Alpha'
  if not isSkipTitle:
    st.header(script)
  #####
  a=.15
  b=.1
  l = ul.spl('TPP,TPP2,IBS,RSS,JMR,SCI,VCA,BTS,GEO')
  d = {
    # Systems
    'TPP': a,  # *** review monthly ***
    'TPP2': a,  # _TPP2
    'IBS': a, # _IBS
    'RSS': a, # _RSS
    'JMR': a,  # _JMR
    #####
    'SCI': a,  # *** review monthly ***
    'VCA': a, # _ETC
    #####
    'BTS': a,  # _ETC
    'GEO': a,  # *** review monthly ***
    #####
    # SAA
    'PFMN.TO': b,
    'BHMG.LSE': b,
  }
  st.write(f"Total weights: {np.sum(list(d.values())):.2f}")
######
#  Calmar: 8.91          MAR: 7.25          Sharpe: 3.81          Cagr: 32.7%          MaxDD: 4.5% before OTS
#    Calmar: 9.11          MAR: 7.43          Sharpe: 3.89          Cagr: 33.4%          MaxDD: 4.5% after OTS
#####
  tickers = d.keys() - l
  dp, dw, _, _ = btSetup(tickers,yrStart=yrStart-1)
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

  if IS_CORRS:
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
  runIBS(y)
  st.divider()
  runRSS(y)
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
  runGEO(y)
  st.divider()

runAlpha(y, isSkipTitle=True)

