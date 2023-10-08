##############
# Util Library
##############
import streamlit as st
import pandas as pd
import numpy as np
import os
import pendulum
import filelock
import json
import termcolor

###########
# Streamlit
###########
def stPageConfig(title):
  st.set_page_config(page_title=title)
  st.title(title)

def stRed(label, z):
  st.markdown(f"{label}: <font color='red'>{z}</font>", unsafe_allow_html=True)

#######
# Cache
#######
def cacheMemory(mode, key, value=None):
  if not hasattr(cacheMemory, 'd'):
    cacheMemory.d=dict()
  if mode=='w':
    cacheMemory.d[key]=value
  elif mode=='r':
    try:
      return cacheMemory.d[key]
    except:
      return None

######
# JSON
######
def jSetFFN(ffn):
  cacheMemory('w','json_ffn',ffn)

def jDump(key, value):
  jDict = jLoadDict()
  jDict[key] = value
  jDumpDict(jDict)

def jDumpDict(jDict):
  ffn=cacheMemory('r','json_ffn')
  with filelock.FileLock(f"{ffn}.lock"):
    with open(ffn, 'w') as f:
      json.dump(jDict, f)

def jEmpty():
  jDumpDict(dict())

def jLoad(key):
  return jLoadDict().get(key, np.nan)

def jLoadDict():
  ffn=cacheMemory('r','json_ffn')
  if os.path.exists(ffn):
    with filelock.FileLock(f"{ffn}.lock"):
      with open(ffn) as f:
        jDict=json.load(f)
  else:
    jDict=dict()
  return jDict

####################################################################################################

def colored(z, color=None, on_color=None, attrs=None):
  return termcolor.colored(z, color=color, on_color=on_color, attrs=attrs)

def getCurrentTime(isCondensed=False):
  return pendulum.now().format(f"{'' if isCondensed else 'YYYY-MM-DD'} HH:mm:ss")

def merge(*args,how='inner'):
  df=args[0]
  for i in range(1,len(args)):
    df2=args[i]
    df=pd.merge(df,df2,how=how,left_index=True,right_index=True)
  return df

def printDict(d, indent=0, isSort=True):
  keys=d.keys()
  if isSort:
    keys=sorted(keys)
  for key in keys:
    value=d[key]
    print('\t' * indent + str(key))
    if isinstance(value, dict):
      printDict(value, indent + 1, isSort=isSort)
    else:
      print('\t' * (indent + 1) + str(value))

def printHeader(header='',isCondensed=False,isAddTime=False, color=None):
  if not isCondensed: print()
  print('-' * 100)
  if not isCondensed: print()
  if len(header) > 0:
    z=f"[{header}{'' if not isAddTime else f' - {getCurrentTime()}'}]"
    if color is None:
      print(z)
    else:
      print(colored(z,color))
    if not isCondensed: print()