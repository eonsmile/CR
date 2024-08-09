##############
# Util Library
##############
import streamlit as st
import pandas as pd
import pathlib
import os
import pendulum
import time
import colorama
from filelock import FileLock
from joblib import Parallel, delayed
import pretty_errors # keep here

#########
# Globals
#########
colorama.init()

###########
# Constants
###########
CACHE_DIR = "c:/cache" if os.getenv('OS', '').startswith('Win') else pathlib.Path(os.path.dirname(__file__))

###########
# Streamlit
###########
def isSt():
  try:
    instance = st.runtime.get_instance()
    return True if instance else False
  except:
    return False

def stCheckPW(key):
  def m():
    isPWOk=st.session_state['pw'] == st.secrets[key]
    st.session_state['isPWOk']=isPWOk
  #####
  def m2():
    st.text_input('Password', type='password', on_change=m, key='pw')
  #####
  if 'isPWOk' not in st.session_state:
    m2()
    return False
  elif not st.session_state['isPWOk']:
    m2()
    st.error('😕 Password incorrect')
    return False
  else:
    return True

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

def cachePersist(mode, key, value=None, expireMins=1e9):
  if mode=='r' and expireMins is None: iExit(f"cachePersist (mode=='{mode}'; key=='{key}'; expireMins=='{expireMins}')")
  path = pathlib.Path(CACHE_DIR)
  if not path.exists(): iExit(f"cachePersist (path=='{path}')")
  ffn = path / f"{key}.pickle"
  with FileLock(f"{ffn}.lock"):
    if mode=='w':
      pd.to_pickle(value, ffn, compression='infer')
    elif mode=='r':
      if (not ffn.exists()) or (time.time() - ffn.lstat().st_mtime >= 60*expireMins): return None
      return pd.read_pickle(ffn)
    else:
      iExit(f"cachePersist (mode=='{mode}')")

#####
# Etc
#####
def colored(text, color=None, isReverse=False, isUnderline=False):
  if not hasattr(colored, 'isInit'):
    colored.isInit=True
    colorama.init()
  z = text
  if color is not None:
    z = f"{getattr(colorama.Fore, color.upper())}{z}"
  if isReverse:
    if color is None: color='WHITE'
    z=f"{getattr(colorama.Fore, 'BLACK')}{getattr(colorama.Back, color.upper())}{text}"
  if isUnderline:
    z=f"\033[4m{z}"
  return f"{z}{colorama.Style.RESET_ALL}"

def getCurrentTime(isCondensed=False):
  return pendulum.now().format(f"{'' if isCondensed else 'YYYY-MM-DD'} HH:mm:ss")

def iExit(msg, isSuffix=True):
  z=f"{msg}{' is invalid!' if isSuffix else ''}"
  if isSt():
    stRed('Abnormal termination', z)
  else:
    tcPrint(z,'red')
  os._exit(1)

def invertDict(d):
  return {v: k for k, v in d.items()}

def merge(*args,how=None):
  if how is None: iExit('merge (how)')
  df=args[0]
  for i in range(1,len(args)):
    df2=args[i]
    df=pd.merge(df,df2,how=how,left_index=True,right_index=True)
  return df

def parallelRun(objs, n_jobs=-1):
  Parallel(n_jobs=n_jobs, backend='threading')(delayed(obj.run)() for obj in objs)

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

def printHeader(header='',isCondensed=False,isAddTime=False):
  if not isCondensed: print()
  print('-' * 100)
  if not isCondensed: print()
  if len(header) > 0:
    z=f"[{header}{'' if not isAddTime else f' - {getCurrentTime()}'}]"
    print(z)
    if not isCondensed: print()

def spl(z):
  return [] if z == '' else z.split(',')

def timeTag(z):
  return f"{getCurrentTime()}: {z}"

def tPrint(z, end='\n'):
  print(timeTag(z),end=end)

def tcPrint(z, color, end='\n'):
  print(timeTag(colored(z, color)), end=end)
