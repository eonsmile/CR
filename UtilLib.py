##############
# Util Library
##############
import pandas as pd
import numpy as np
import os
import filelock
import termcolor
import json

#######
# Cache
#######
# Cache items in memory
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

# Dump a variable into json file
def jDump(key, value):
  jDict = jLoadDict()
  jDict[key] = value
  jDumpDict(jDict)

# Dump dict into json file
def jDumpDict(jDict):
  ffn=cacheMemory('r','json_ffn')
  with filelock.FileLock(f"{ffn}.lock"):
    with open(ffn, 'w') as f:
      json.dump(jDict, f)

# Empty out json file
def jEmpty():
  jDumpDict(dict())

# Load a variable from json file
def jLoad(key):
  return jLoadDict().get(key, np.nan)

# Load dict from json file
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

# Merge two dataframes or series
def merge(*args,how='inner'):
  df=args[0]
  for i in range(1,len(args)):
    df2=args[i]
    df=pd.merge(df,df2,how=how,left_index=True,right_index=True)
  return df

# Print dictionary
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

# Print header
def printHeader(header='',isCondensed=False,color=None):
  if not isCondensed: print()
  print('-' * 100)
  if not isCondensed: print()
  if len(header) > 0:
    z=f"[{header}]"
    if color is None:
      print(z)
    else:
      print(termcolor.colored(z,color))
    if not isCondensed: print()
