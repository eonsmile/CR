##############
# Util Library
##############
import streamlit as st
import pandas as pd
import pathlib
import os
import pendulum
import pickle
import time
import termcolor
from filelock import FileLock

###########
# Constants
###########
CACHE_DIR = "c:/cache" if os.getenv('OS', '').startswith('Win') else pathlib.Path(os.path.dirname(__file__))

###########
# Streamlit
###########
def stRed(label, z):
  st.markdown(f"{label}: <font color='red'>{z}</font>", unsafe_allow_html=True)

def stWriteDf(df,isMaxHeight=False):
  df2 = df.copy()
  if isinstance(df2.index, pd.DatetimeIndex):
    df2.index = pd.to_datetime(df2.index).strftime('%Y-%m-%d')
  if isMaxHeight:
    height = (len(df2) + 1) * 35 + 3
    st.dataframe(df2, height=height)
  else:
    st.write(df2)

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

def cachePersist(mode, key, value=None, expireMins=1e9):
  if mode=='r' and expireMins is None: iExit(f"cachePersist (mode=='{mode}'; key=='{key}'; expireMins=='{expireMins}')")
  path = pathlib.Path(CACHE_DIR)
  if not path.exists(): iExit(f"cachePersist (path=='{path}')")
  ffn = path / f"{key}.pickle"
  with FileLock(f"{ffn}.lock"):
    if mode=='w':
      with ffn.open('wb') as f: pickle.dump(value, f)
    elif mode=='r':
      if (not ffn.exists()) or (time.time() - ffn.lstat().st_mtime >= 60*expireMins): return None
      with ffn.open('rb') as f: return pickle.load(f)
    else:
      iExit(f"cachePersist (mode=='{mode}')")

#####
# Etc
#####
def colored(z, color=None, on_color=None, attrs=None):
  return termcolor.colored(z, color=color, on_color=on_color, attrs=attrs)

def getCurrentTime(isCondensed=False):
  return pendulum.now().format(f"{'' if isCondensed else 'YYYY-MM-DD'} HH:mm:ss")

def iExit(msg, isSuffix=True):
  z=f"{msg}{' is invalid!' if isSuffix else ''}"
  stRed('Abnormal termination',z)
  os._exit(1)

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

def spl(z):
  return [] if z == '' else z.split(',')
