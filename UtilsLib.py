###############
# Utils Library
###############
import pathlib
import numpy as np
import os
import filelock
import json

# Params
fn='CR.json'

#############################################################################################

######
# JSON
######
# Get full filname for a data file
def getDataFN(fn):
  return pathlib.Path(os.path.dirname(__file__)) / fn

# Dump a variable into json file
def jDump(key, value, filename):
  jDict = jLoadDict(filename)
  jDict[key] = value
  jDumpDict(jDict, filename)

# Dump dict into json file
def jDumpDict(jDict,filename):
  fn=getDataFN(filename)
  with filelock.FileLock(f"{fn}.lock"):
    with open(fn, 'w') as f:
      json.dump(jDict, f)

# Empty out json file
def jEmpty(filename):
  jDumpDict(dict(),filename)

# Load a variable from json file
def jLoad(key,filename):
  return jLoadDict(filename).get(key, np.nan)

# Load dict from json file
def jLoadDict(filename):
  fn=getDataFN(filename)
  if os.path.exists(fn):
    with filelock.FileLock(f"{fn}.lock"):
      with open(fn) as f:
        jDict=json.load(f)
  else:
    jDict=dict()
  return jDict

####################################################################################################

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
def printHeader(header=''):
  print()
  print('-' * 100)
  print()
  if len(header) > 0:
    print('['+header+']')
    print()
