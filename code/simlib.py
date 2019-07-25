# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:57:51 2014

@author: dalbis
"""
import sys
import os
from time import clock
from numpy import floor,remainder,float64


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False
        
        
    
def format_elapsed_time(delta_clock):
  """
  Format elapsed time in a human readable format
  """
  hours = floor(delta_clock/3600)
  minutes = floor(delta_clock/60-hours*60)
  seconds = floor(delta_clock-hours*3600-minutes*60)
  
  string=''
  if hours>0:
     string+=' %dh'%hours
  if minutes>0:
     string+=' %dm'%minutes
     
  string+=' %ds'%seconds
  return string
  
def print_progress(snap_idx,num_snaps,start_clock=None,step=None):
  """
  Prints a progress bar to the console
  """
  
  if step is None:
    step=num_snaps/100.0
    
  if snap_idx>0:
    snap_idx+=1
    if remainder(snap_idx,float(step))==0:
      progress = int(snap_idx/float(num_snaps)*100)
      string = '\r[{:20s}] {:3d}% complete'.format('#'*(progress/5), progress)
    
      if progress>0 and start_clock is not None:
        cur_clock = clock()
        elapsed_time = cur_clock-start_clock
        remaining_time = (elapsed_time/progress)*(100-progress)
        string+=',%s elapsed, about%s remaining' %(format_elapsed_time(elapsed_time),format_elapsed_time(remaining_time))
  
      if progress == 100:
        print string
      else:
        print string,
        
      if hasattr(sys.stdout,'flush'):
        sys.stdout.flush()
          
def sendResults(simId,simTitle,logString,figurePaths):
  """
  Sends an e-mail with the simulation results
  """

  
  import smtplib
  import email
  import email.mime.application

  # Create a text/plain message
  msg = email.mime.Multipart.MIMEMultipart()
  msg['Subject'] = 'Simulation results: %s'%simTitle
  msg['From'] = 'tiziano.dalbis@cms.hu-berlin.de'
  msg['To'] = 'tizyweb@gmail.com'

  # The main body is just another attachment
  body = email.mime.Text.MIMEText(logString)
  msg.attach(body)

  # Add attachments
  if isinstance(figurePaths, basestring):
    figurePaths=[figurePaths]
    
  for figurePath in figurePaths:
    fp=open(figurePath,'rb')
    att = email.mime.application.MIMEApplication(fp.read(),_subtype="png")
    fp.close()
    att.add_header('Content-Disposition','attachment',filename=figurePath)
    msg.attach(att)

  # send
  s = smtplib.SMTP('mailhost.cms.hu-berlin.de',port=587)
  s.starttls()
  s.login('dalbisti','Brainstem2014.')
  s.sendmail(msg['From'],msg['To'] , msg.as_string())
  s.quit()

def ensureParentDir(path):
  """
  Ensure that a given path exists
  """
  parentDir = os.path.realpath(path+'/..')
  if not os.path.exists(parentDir):
    os.makedirs(parentDir)

def ensureDir(path):
  if not os.path.exists(path):
    os.makedirs(path)
  
def gen_hash_id(string):

  import hashlib  
  hash_object = hashlib.md5(string.encode())
  return str(hash_object.hexdigest())

def gen_string_id(paramMap,key_params=None):
  str_id=''
  if key_params is None:
    keys=paramMap.keys()
  else:
    keys=key_params
  for key in keys:
    str_id+='%s=%s_'%(key,format_val(paramMap[key])) 
  str_id=str_id[:-1]
  return str_id



def format_val(val):
  if (type(val)==float64 or type(val)==float) and (abs(val)<1e-3 or abs(val)>1e3):
    val_str='%.3e'%val 
  else:
    val_str=str(val)
     
  return val_str
    
def params_to_str(paramMap,keyParams=None,compact=False,to_exclude=[]):
  
  if keyParams is None:
    keys=paramMap.keys()
  else:
    keys=keyParams
    
  if compact is False:
    logStr='\n'
    logStr+='========== PARAMETERS ==========\n'
    delimiter='\n'
    equal=' = '
  else:
    logStr=''
    delimiter=', '
    equal='='
  for key in sorted(keys):
    if key not in to_exclude:
     val=paramMap[key]
     val_str=format_val(val)
     logStr+=key+equal+val_str+delimiter
     
  if compact is False:    
    logStr+='\n'
  else:
    logStr=logStr[0:-len(delimiter)]
  return logStr    
    
def logSim(simId,simTitle,tsStr,teStr,elapsedTime,paramMap,paramsPath,doPrint=True):
  """
  Logs simulation parameters and simulation time to a string.
  The string is also saved to a text file stored in paramsPath
  """
  import socket

  logStr=''
  logStr+='Simulation Title: %s \n'%simTitle
  logStr+='Simulation Id: %s \n' %simId
  logStr+='Running on: %s\n'% socket.gethostname()
  logStr+= 'Simulation started: %s\n'%tsStr
  logStr+= 'Simulation ended: %s\n'%teStr
  logStr+='Elapsed time: %s \n' %format_elapsed_time(elapsedTime)
  logStr+=params_to_str(paramMap)
  if doPrint is True:
    print logStr
  ensureParentDir(paramsPath)
  f=open(paramsPath,'w')
  f.write(logStr)
  f.close()
  return logStr
  
class Tee(object):
  """
  Class for duplicating stdout to log file
  """  
  def __init__(self, name):
      self.file = open(name, 'w')
      self.stdout = sys.stdout
      sys.stdout = self
  def __del__(self):
      sys.stdout = self.stdout
      self.file.close()
  def write(self, data):
      self.file.write(data)
      self.file.flush()
      self.stdout.write(data)
      self.stdout.flush()

def get_unique_path(baseName):
  """
  Returns a unique path name
  """  

  suffix=''
  idx=1
  while True:
    if not os.path.exists(baseName+suffix):
      return baseName+suffix
    else:
      suffix='_%d'%idx
      idx+=1
