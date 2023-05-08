import streamlit as st
import UtilsLib as ul
import datetime
import numpy as np
import pandas as pd
from PIL import Image

###########
# Constants
###########
DT_FORMAT='%d%b%y'

######
# Main
######
st.set_page_config(page_title='Core Reporter')
st.title('Core Reporter')

# Weights
st.header('Weights')
lastUpdateDict=ul.jLoad('lastUpdateDict')
dts=[]
for v in lastUpdateDict.values():
  dts.append(datetime.datetime.strptime(v, DT_FORMAT))
f=lambda dt: datetime.datetime.strftime(dt,DT_FORMAT)
lastUpdate=f(np.max(dts))
st.write(f"Last Update: {lastUpdate}")
dts=[f(dt) for dt in dts]

l=list()
d=ul.jLoad('IBSDict')
ep=1e-9
ibsDict={'SPY':0,
         'QQQ':d['QQQ']+ep,
         'TLT':d['TLT']+ep,
         'IEF':0,
         'GLD':0,
         'UUP':0}
d=ul.jLoad('TPPDict')
tppDict={'SPY':d['SPY']+ep,
         'QQQ':d['QQQ']+ep,
         'TLT':0,
         'IEF':d['IEF']+ep,
         'GLD':d['GLD']+ep,
         'UUP':d['UUP']+ep}
i=0
for und in ['SPY','QQQ','TLT','IEF','GLD','UUP']:
  l.append([dts[i],und,(ibsDict[und]+tppDict[und])/2,ibsDict[und],tppDict[und]])
  i+=1
df=pd.DataFrame(l)
df.columns = ['Last Update', 'ETF', 'Total Weight', 'IBS (1/2)', 'TPP (1/2)']
df.set_index(['ETF'],inplace=True)
cols=['Total Weight','IBS (1/2)','TPP (1/2)']
df[cols] = df[cols].applymap(lambda n:'' if n==0 else f"{n:.1%}")
st.dataframe(df.style.apply(lambda row: ['background-color:red'] * len(row) if row['Last Update']==lastUpdate else [''] * len(row), axis=1))

# Realized Performance
st.header('Realized Performance')
lastUpdate2=ul.jLoad('lastUpdateDict2')
st.write(f"Last Update: {lastUpdate2['realizedPerformance']}")
st.write(f"MTD: {ul.jLoad('mtd'):.1%}")
st.write(f"YTD: {ul.jLoad('ytd'):.1%}")

# Backtest - Static
st.header('Backtest - Static')
st.write(f"Last Update: {lastUpdate2['backtestStatic']}")
image = Image.open('BacktestStatic.png')
st.image(image)
st.markdown('YTD figures under **Realized Performance** can be different to those under **Backtest - Static** because of model changes implemented since the beginning of the year.')

# Backtest - Live
st.header('Backtest - Live')
st.markdown('[Link](https://colab.research.google.com/drive/1dLe29LuqDhMy_MaIgl-OApsuIBwx3kCE?usp=sharing#forceEdit=true&sandboxMode=true)')

# Beta
st.header('Beta (Return regressions of futures vs. ETFs)')
st.markdown('[Link](https://colab.research.google.com/drive/1gIIpQyPfEUp5tEUvljG6KAvd_CcruRL3?usp=sharing#forceEdit=true&sandboxMode=true)')