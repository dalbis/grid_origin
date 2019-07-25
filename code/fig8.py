# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:54:58 2016

@author: dalbis
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:30:07 2016

@author: dalbis
"""


from grid_params import GridRateParams
import pylab as pl
import numpy as np
from grid_functions import map_merge
from grid_const import ModelType
import plotlib as pp
from grid_batch import GridBatch
from simlib import run_from_ipython
from simlib import ensureDir
import gridlib as gl

figures_path='../figures'
ensureDir(figures_path)


num_seeds=200


a=4.
J_av_star=0.05
sigmas=[0.0625,0.0625]
tau2s=[0.16,0.35]
input_means=[0.3,0.1]
batches=[]
for sigma,tau2,input_mean in zip(sigmas,tau2s,input_means):

  param_map=map_merge(  GridRateParams.gau_grid_large_arena_biphasic_neg,
      {
      'dt':50.,
      'r0':10.,
      'compute_scores':False,
      'sigma':sigma,
      'tau2':tau2,
      'input_mean':input_mean,
      'a':a
      })    


  print 'sigma: %.4f tau2: %.3f r_av=%.3f  '%(sigma,tau2,input_mean)


  batch=GridBatch(ModelType.MODEL_RATE_AVG,param_map,{'seed':np.arange(num_seeds)})

  
  do_run=batch.post_init()
  batches.append(batch)
  
  if do_run and not run_from_ipython():
    batch.run()
    batch.post_run()
  

 

#%%

## COLLECT DATA

batch_angles=[]
batch_phases=[]
batch_final_weights_mat=[]
batch_spacings=[]
batch_max_freqs=[]
num_grids=[]
suffix=['a','b'] 

for batch_idx in xrange(2):

  batch=batches[batch_idx]
  batch_data=np.load(batch.batch_data_path)  
  final_weights_map=batch_data['final_weights_map'][()]
  
  spacings_map=batch_data['final_weight_spacing_map'][()]
  angles_map=batch_data['final_weight_angle_map'][()] 
  phases_map=batch_data['final_weight_phase_map'][()]
  scores_map=batch_data['final_weight_score_map'][()]
  idxs=np.array(scores_map.values())>0.5
  print '%d/%d with score higher than 0.5'%(sum(idxs),num_seeds)
  
  spacings=np.array(spacings_map.values())[idxs]
  angles=np.array(angles_map.values())[idxs]
  phases=[]
  num_idxs=np.where(idxs)[0]
  for idx in num_idxs:
    phase=phases_map.values()[idx].tolist()
    if type(phase)==list:
      phases.append(phase)
  phases=np.array(phases)
  batch_phases.append(phases)
  # compute max_freqs
  final_weights_mat=np.array(final_weights_map.values())[idxs]
  final_weights_mat=np.swapaxes(final_weights_mat,0,2)
  final_weights_mat=np.swapaxes(final_weights_mat,0,1)
  weigts_dfts,freqs,allfreqs=gl.dft2d_num(final_weights_mat,param_map['L'],param_map['n'])
  profiles=gl.dft2d_profiles(weigts_dfts)
  max_freqs=freqs[np.argmax(profiles,axis=1)]

  num_grids.append(idxs.sum())
  batch_angles.append(angles)
  batch_spacings.append(spacings)
  batch_max_freqs.append(max_freqs)
  batch_final_weights_mat.append(final_weights_mat)
  
#%%

## PLOT PHASES AND ANGLES

pl.rc('font',size=10)
pp.set_tick_size(3)

for batch_idx in xrange(2):

  
  # phases
  
  pl.figure(figsize=(1.7,1.7))
  pl.subplots_adjust(left=0.4,right=0.9,top=0.85,bottom=0.3,wspace=0.2,hspace=0.6)
  
  pl.subplot(1,1,1,aspect='equal')
  pl.plot(batch_phases[batch_idx][:,0],batch_phases[batch_idx][:,1],'.k',ms=7)
  pp.custom_axes()
  pl.xlabel('Grid phase-x [m]')
  pl.ylabel('Grid phase-y [m]')
  pl.xlim(-0.3,0.3)
  pl.ylim(-0.3,0.3)
  pl.xticks([-0.3,0.,0.3])
  pl.yticks([-0.3,0.,0.3])
  pl.savefig(figures_path+'/fig8'+suffix[batch_idx]+'1.eps', dpi=300,transparent=True)
  
  
  # angles
  pl.figure(figsize=(4.,1.5))  
  pl.subplots_adjust(left=0.3,right=1,top=0.85,bottom=0.3,wspace=0.2)
  
  pl.subplot(1,3,1)
  bins,edges=np.histogram(np.array(batch_angles[batch_idx])*180/np.pi,range=[0,60],bins=30)
  pl.bar(edges[:-1],bins.astype(np.float)/sum(idxs),width=edges[1]-edges[0],color='k')
  pp.custom_axes()
  pl.xlim(0,60)
  pl.xlabel('Grid Orientation [degrees]')
  pl.ylabel('Fraction of grids')
  pl.xticks([0,30,60])
  pl.yticks([0,0.6])
  pl.ylim(0,0.6)
  pl.savefig(figures_path+'/fig8'+suffix[batch_idx]+'2.eps', dpi=300,transparent=True)
  

#%%


## PLOT EXAMPLE SPATIAL MAPS 

pl.figure()
pl.subplot(1,2,1,aspect='equal')
pl.pcolormesh(batch_final_weights_mat[0][:,:,3])
pp.noframe()

pl.subplot(1,2,2,aspect='equal')
pl.pcolormesh(batch_final_weights_mat[0][:,:,2])
pp.noframe()
pl.savefig(figures_path+'/fig8a3.eps', dpi=300,transparent=True)


pl.figure()
pl.subplot(1,2,1,aspect='equal')
pl.pcolormesh(batch_final_weights_mat[1][:,:,2])
pp.noframe()

pl.subplot(1,2,2,aspect='equal')
pl.pcolormesh(batch_final_weights_mat[1][:,:,3])
pp.noframe()
pl.savefig(figures_path+'/fig8b3.eps', dpi=300,transparent=True)

#%%

