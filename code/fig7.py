# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:30:07 2016

@author: dalbis
"""


from multiprocessing import Pool
import socket
from grid_params import GridRateParams
import pylab as pl
import numpy as np
from grid_functions import map_merge,get_params,load_data
from grid_rate_avg import GridRateAvg
from grid_rate import GridRate
from grid_const import ModelType
import plotlib as pp
import traceback 
import os
from simlib import ensureDir


figures_path='../figures'
ensureDir(figures_path)


def function(sim):

  try:
    sim.post_init(do_print=False)
    sim.run(do_print=False)
    sim.post_run(do_print=False)  
  
  except Exception:
    print
    print 'Exception'
    traceback.print_exc()
  
    
# maximum processes per host
procs_by_host={'compute1':20, 'compute2':20, 'compute3':10,'cluster01':10, 'mcclintock':7}


model_type=ModelType.MODEL_RATE_AVG


a=4.0
J_av_star=0.05

sigmas=[0.045,0.045,0.0625,0.0625]
tau2s=[0.16,0.35,0.16,0.35]
input_means=np.array([ 0.21 ,  0.085,  0.30 ,  0.10 ])
seeds=[0,1,0,0]

final_weight_maps=[]

# create pool  
host=socket.gethostname()  
num_procs=procs_by_host[host]
pool=Pool(processes=num_procs)
sims=[]
hashes=[]
paramMaps=[]  
  
rate_avg_params=map_merge(
    GridRateParams.gau_grid_large_arena_biphasic_neg,
    {
    'dt':50.,
    'compute_scores':False,
    })
    
      
    
for sigma,tau2,input_mean,seed in zip(sigmas,tau2s,input_means,seeds):
  
  paramMap=map_merge(rate_avg_params,
    {
    'sigma':sigma,
    'input_mean':input_mean,
    'tau2':tau2,
    'seed':seed,
    })
    
    
  p=get_params(paramMap)
  paramMap['amp']=p.input_mean*p.L**2/(2*np.pi*p.sigma**2)
  paramMap['b1']=1/p.tau1
  paramMap['b2']=1/p.tau2
  paramMap['b3']=1/p.tau3
  

  paramMap['a']=a
  
  
  paramMaps.append(paramMap)

    
  if model_type==ModelType.MODEL_RATE_AVG:
    sim = GridRateAvg(paramMap)
    print 'RATE AVG'
  else:
    sim = GridRate(paramMap)
      
  print 'sigma: %.4f tau2: %.3f'%(sigma,tau2)
  print 'hash_id: %s'%sim.hash_id
  
  if sim.do_run is True:
    print 'ADDING TO POOL'
    sims.append(sim)
  else:  
    print 'PRESENT'
    print sim.dataPath
  hashes.append(sim.hash_id)



# run
if len(sims)>0:
  
  print 'BATCH MODE: Starting %d/%d processes on %s'%(len(sims),num_procs,host)
  
  for sim in sims:
    pool.apply_async(function,args=(sim,)) 
  
  pool.close()
  pool.join() 
   
 
  
#%%

# PLOT THE GRID MAPS

if model_type==ModelType.MODEL_RATE_AVG:
  results_path=GridRateAvg.results_path
else:
  results_path=GridRate.results_path


pl.figure(figsize=(8.5,3))
pl.subplots_adjust(hspace=0.3,wspace=0.6)


for idx,hash_id in enumerate(hashes):
          
  dataPath=os.path.join(results_path,'%s_data.npz'%hash_id)
  p,r=load_data(dataPath)
  pl.subplot(2,2,idx+1,aspect='equal')  
  print 'sigma: %.4f  tau_L=%.3f'%(p.sigma,p.tau2)
  print 'max_freq: %.3f'%p.max_freq
  print 'max eig: %.3f'%p.max_eig
  print 'max_weight %.2f'%r.final_weights.max()
  print 'tau_str=%e'%p.tau_str
  print 'tau_av=%e'%p.tau_av
  print 'B=%.2f'%p.B
  print 

  pl.pcolormesh(r.final_weights,rasterized=True)
  pp.noframe()

  
pl.savefig(figures_path+'/fig7_maps.eps',bbox_inches='tight',dpi=300)

#%%


# PLOT THE FOURIER SPECTRA OF THE GRID MAPS

pl.figure(figsize=(8.5,3))
pl.subplots_adjust(hspace=0.3,wspace=0.6)
for idx,hash_id in enumerate(hashes):
  
    
  dataPath=os.path.join(results_path,'%s_data.npz'%hash_id)
  p,r=load_data(dataPath)
  print p.max_freq  
  print '%.2f'%p.B
  pl.subplot(2,2,idx+1,aspect='equal')   
  pp.plot_matrixDFT(r.final_weights,p.L/p.n,circle_radius=p.max_freq,cmap='binary',circle_color=pp.red,lw=2)
  pl.xlim(-5,5)  
  pl.ylim(-5,5)  
  pl.xticks([])
  pl.yticks([])
  
pl.savefig(figures_path+'/fig7_spectra.eps',bbox_inches='tight',dpi=300)  
  
#%%

### PLOT THE TWO INPUT GAUSSIAN TUNING CURVES

# small gaussian
sigma=sigmas[0]
input_mean=input_means[0]
p=get_params(paramMaps[0])

amp=input_mean*p.L**2/(2*np.pi*sigma**2)
pl.figure(figsize=(2.0,1.5))
pl.subplots_adjust(left=0.2,bottom=0.2)
x=np.arange(-1,1,0.001)
pl.plot(x,amp*np.exp(-x**2/(2*sigma**2)),'-k',lw=2)  
pp.custom_axes()
pl.xlim(-0.3,0.3)
pl.ylim(-1,80)
pl.xticks([-0.2,0,0.2])
pl.yticks([0,30])
pl.savefig(figures_path+'/fig7_gau_small.eps')

# large gaussian
sigma=sigmas[2]
input_mean=input_means[2]

amp=input_mean*p.L**2/(2*np.pi*sigma**2)
pl.figure(figsize=(2.0,1.5))
pl.subplots_adjust(left=0.2,bottom=0.2)

x=np.arange(-1,1,0.001)
pl.plot(x,amp*np.exp(-x**2/(2*sigma**2)),'-k',lw=2)  
pp.custom_axes()
pl.xlim(-0.3,0.3)
pl.ylim(-1,80)
pl.xticks([-0.2,0,0.2])
pl.yticks([0,30])
pl.savefig(figures_path+'/fig7_gau_large.eps')

#%%


### PLOT THE TWO FILTERS

from grid_functions import K_t

#short filter
p=get_params(paramMaps[0])
pl.figure(figsize=(2.0,2.))
pl.subplots_adjust(left=0.2,bottom=0.2)
t=np.arange(0,3,0.001)
pl.plot(t,K_t(p.b1,p.b2,p.b3,p.mu1,p.mu2,p.mu3,t),'-k',lw=2)
pp.custom_axes()
pl.xlim(0,1.5)
pl.ylim(-2,6)
pl.xticks([0,0.5])
pl.yticks([0,2])
pl.savefig(figures_path+'/fig7_filt_short.eps')

#long filter
p=get_params(paramMaps[1])
pl.figure(figsize=(2.0,2.))
pl.subplots_adjust(left=0.2,bottom=0.2)
t=np.arange(0,3,0.001)
pl.plot(t,K_t(p.b1,p.b2,p.b3,p.mu1,p.mu2,p.mu3,t),'-k',lw=2)
pp.custom_axes()
pl.xlim(0,1.5)
pl.ylim(-2,6)
pl.xticks([0,0.5])
pl.yticks([0,2])
pl.savefig(figures_path+'/fig7_filt_long.eps')
