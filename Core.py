######
# Core
######
import sys
import os
if 'OS' in os.environ and os.environ['OS'].startswith('Win'): sys.path.insert(1, 'c:/onedrive/py4/')

import UtilLib as ul
import QuantLib as ql
import streamlit as st
import numpy as np
import pandas as pd

########
# Params
########
yrStart=2011
strategies = ['IBS', 'TPP']
weights = [1/2,1/2]

######
# Init
######
script='Core'
st.title(script)

######
# Main
######
# Weights
st.header('Weights')
z=zip(strategies,weights)
df=pd.DataFrame(z,columns=['Strategy','Weight']).set_index('Strategy')
ql.stWriteDf(df)

# Calcs
d=ul.jLoadDict()
dp=pd.DataFrame()
for strategy in strategies:
  dp[strategy]=pd.read_json(d[strategy],typ='series')
dp=ql.applyDates(dp, dp[strategies[1]]).fillna(method='pad')
dw=dp*np.nan
pe=ql.endpoints(dw, 'M')
for i in range(len(weights)):
  dw[strategies[i]].iloc[pe]=weights[i]

# Backtest
ql.bt(script, dp, dw, yrStart)

# Recent performance
st.header('Recent Performance')
dp2=dp.copy()
dp2[script]=pd.read_json(ul.jLoad(script), typ='series')
dp2=dp2[[script]+strategies]
dp2=round((dp2/dp2.iloc[-1]).tail(23)*100,2)
ql.stWriteDf(dp2,isMaxHeight=True)