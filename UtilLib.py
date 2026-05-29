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
import pickle
import dill
import threading
import pretty_errors # keep here

#########
# Globals
#########
colorama.init()

###############
# Caching/Locks
###############
CACHE_DIR = "c:/cache" if os.getenv('OS', '').startswith('Win') else pathlib.Path(os.path.dirname(__file__))
CACHE_PATH = pathlib.Path(CACHE_DIR)
pickle.DEFAULT_PROTOCOL = 5
tLocks = dict()
_tLocksGuard = threading.Lock()

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

def cachePersist(mode, key, value=None, expireMins=1e9, isDill=False):
  if mode=='r' and expireMins is None: iExit(f"cachePersist: mode '{mode}'; key '{key}'; expireMins '{expireMins}'")
  fkey = f"{key}.{'dill' if isDill else 'pickle'}"
  ffn = CACHE_PATH / fkey
  with fLock(fkey, timeout=300):
    if mode=='w':
      if isDill:
        with open(ffn, 'wb') as f: dill.dump(value, f)
      else:
        pd.to_pickle(value, ffn, compression='infer')
    elif mode=='r':
      if (not ffn.exists()) or (time.time() - ffn.lstat().st_mtime >= 60*expireMins): return None
      try:
        if isDill:
          with open(ffn, 'rb') as f: r=dill.load(f)
        else:
          r=pd.read_pickle(ffn)
      except Exception as err:
        tcPrintErr('cachePersist', err=err)
        r=None
      return r
    else:
      iExit(f"cachePersist: mode '{mode}'")

def fLock(key, timeout=-1):
  return FileLock(f"{CACHE_PATH / key}.lock", timeout=timeout)

class _sLock:
  def __init__(self, key): self.key = key
  def __enter__(self):
    self.t = tLock(self.key); self.t.acquire()
    try:
      self.f = fLock(self.key); self.f.acquire()
    except:
      self.t.release(); raise
    return self
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.f.release(); self.t.release()

def sLock(key):
  return _sLock(key)

def tLock(key):
  if key not in tLocks:
    with _tLocksGuard:
      if key not in tLocks:
        tLocks[key] = threading.Lock()
  return tLocks[key]

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

'''
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
'''

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

def cPrint(z, color, isReverse=False, isUnderline=False, end='\n'):
  print(colored(z, color, isReverse=isReverse, isUnderline=isUnderline),end=end)

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

def pushoverSend(msg):
  import http.client, urllib.parse, time, os
  from dotenv import load_dotenv
  #####
  load_dotenv()
  PUSHOVER_USER = os.getenv('PUSHOVER_USER', '')
  PUSHOVER_TOKEN = os.getenv('PUSHOVER_TOKEN', '')
  data = urllib.parse.urlencode({
    'token': PUSHOVER_TOKEN, 'user': PUSHOVER_USER,
    'message': msg, 'priority': 0, 'sound': 'updown'})
  hdr = {'Content-type': 'application/x-www-form-urlencoded'}
  for i in range(30):
    try:
      c = http.client.HTTPSConnection('api.pushover.net:443')
      c.request('POST', '/1/messages.json', data, hdr)
      c.getresponse()
      return
    except Exception as e:
      print('Pushover error:', e)
      time.sleep(i+1)

def speak(text):
  try:
    import win32com.client as wincl
    speaker = wincl.Dispatch('SAPI.SpVoice')
    speaker.Voice = speaker.getvoices()[1]
    speaker.Speak(text)
  except:
    tPrint(f"[Speaking:'{text}']")
    print()

def spl(z):
  return [] if z == '' else z.split(',')

def timeTag(z):
  return f"{getCurrentTime()}: {z}"

def tPrint(z, end='\n'):
  print(timeTag(z),end=end)

def tcPrint(z, color, end='\n'):
  print(timeTag(colored(z, color)), end=end)

def tcPrintErr(z, err=None, color='red', isPushover=False, isSpeak=False):
  if isinstance(err, Exception):
    msg = f"{z} {type(err).__name__}: {err}"
  elif err is None:
    msg = z
  else:
    msg = f"{z}: {err}"
  tcPrint(msg, color)
  if isPushover: pushoverSend(msg)
  if isSpeak: speak(msg)
