from QuantLib import *
import UtilLib as ul

###########
# Functions
###########
def runBeta(yrStart, isSkipTitle=False):
  script = 'Beta'
  if not isSkipTitle:
    st.header(script)
  #####
  d = {
    'SPY': 0.25,
    'QQQ': 0.25,
    'DBMF': 0.50,
  }
  st.write(
    f"SPY {d['SPY']:.0%} / QQQ {d['QQQ']:.0%} / DBMF {d['DBMF']:.0%}  | monthly rebal  | "
    f"Total weights: {np.sum(list(d.values())):.2f}"
  )
  #####
  dp, dw, _, _ = btSetup(ul.spl('SPY,QQQ,DBMF'), yrStart=yrStart - 1)
  pe = endpoints(dw)
  for und, w in d.items():
    dw.iloc[pe, dw.columns.get_loc(und)] = w
  #####
  bt(script, dp, dw, yrStart)
######
# Main
######
z = 'Beta Reporter'
st.set_page_config(page_title=z)
st.title(z)

chosenYear = st.radio('Start Year', ['2008', '2016'], index=1)
st.write('')
y = int(chosenYear)
runBeta(y, isSkipTitle=True)
