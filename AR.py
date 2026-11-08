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
    'IBS': a,  # _MR
    'RSS': a,  # _MR
    #####
    # Tier B MRs
    'BBS': b,  # _MR
    'GMR': b,  # _MR
    'HMR': b,  # _HMR — harness: 2828.HK Tier-2 → @b
    'IBS2': b,  # _MR — IWM; exit 0.8|hib TS6; VT 0.255 (ql.bt MDD<15%)
    'JMR': b,  # _MR — harness: EWJ/FXY Tier-2 → @b
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
    'BTS': a,  # _BTS — BTC Tier-1; VT 0.20 / maxWgt 1.5 (MDD~14.6%)
    'GEO': b,  # *** review monthly ***
    'WTS': b,  # _ETC
    'ZBT': b,  # *** review monthly *** — a→b 2026-11-08 (sparse sample; AFR nudge)
  }
  st.write(f"Total weights: {np.sum(list(d.values())):.2f}")
######
#  UPI: 48.09          UPI3y: 67.38          Sharpe: 4.40          Cagr: 44.1%          MaxDD: 4.8% (HMR a→b; BTS@a vt0.20 max1.5; BTC Tier-1; sumW 2.44)
#  UPI: 47.26          UPI3y: 65.85          Sharpe: 4.36          Cagr: 43.2%          MaxDD: 4.8% (ZBT a→b sparse; sumW 2.32)
#  UPI: 48.80          UPI3y: 72.48          Sharpe: 4.45          Cagr: 43.7%          MaxDD: 4.4% (TAA HI N63/2% + vt0.22; sumW 2.32)
#  UPI: 50.28          UPI3y: 77.30          Sharpe: 4.42          Cagr: 42.1%          MaxDD: 4.2% (2026-11-08 17:01; TAA GLD booth+HI + vt0.25; sumW 2.32)

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
