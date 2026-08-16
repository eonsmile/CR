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
  b=.1
  l = ul.spl('TPP,TPP2,IBS,RSS,JMR,SCI,VCA,BTS,COS,GEO')
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
    'COS': a,  # _COS
    'GEO': a,  # *** review monthly ***
    #####
    # SAA
    'PFMN.TO': b,
    'BHMG.LSE': b,
  }
  st.write(f"Total weights: {np.sum(list(d.values())):.2f}")
######
#    Calmar: 9.66          MAR: 7.52          Sharpe: 3.92          Cagr: 33.8%          MaxDD: 4.5%
#  Calmar: 9.89          MAR: 7.80          Sharpe: 4.03          Cagr: 35.1%          MaxDD: 4.5% (COS)
#    Calmar: 9.92          MAR: 7.80          Sharpe: 4.03          Cagr: 35.1%          MaxDD: 4.5% (14Aug, before SCI reduction; ytd16.8; mtd3.3)
#    Calmar: 9.90          MAR: 7.88          Sharpe: 4.00          Cagr: 34.5%          MaxDD: 4.4% (after SCI reduction; ytd16.6; mtd3.4)
#    Calmar: 9.80          MAR: 7.85          Sharpe: 3.98          Cagr: 34.4%          MaxDD: 4.4% (after VCA rule change; ytd 16.6; mtd 3.4)
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
  runCOS(y)
  st.divider()
  runGEO(y)
  st.divider()

runAlpha(y, isSkipTitle=True)

