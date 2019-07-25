# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:01:08 2016

@author: dalbis
"""

from grid_params import GridRateParams
import pylab as pl
import numpy as np
from grid_functions import map_merge,get_params
from grid_const import ModelType
import plotlib as pp
from grid_batch import GridBatch
from simlib import run_from_ipython
from grid_inputs import GridInputs



from simlib import ensureDir

figures_path='../figures'
ensureDir(figures_path)


force=False


batch_default_map=map_merge(
                GridRateParams.gau_mix_small_arena_biphasic_neg,
                {
                'dt':10.,
                'compute_scores':False,
                })
p=get_params(batch_default_map)         
      
batch_override_map = {'inputs_seed':np.arange(100)}
    
batch=GridBatch(ModelType.MODEL_RATE_AVG,batch_default_map,batch_override_map,force=force )
do_run=batch.post_init()

if (force or do_run) and not run_from_ipython():
  batch.run()
  batch.post_run()
  

      
#%% PLOT RATE SCORES ANGLES SPACINGS AND PHASES
      
import gridlib as gl

batch_data=np.load(batch.batch_data_path)  

rates_map=batch_data['final_rates_map'][()]
rate_score_map=batch_data['final_rate_score_map'][()]
rate_spacing_map=batch_data['final_rate_spacing_map'][()]
rate_angle_map=batch_data['final_rate_angle_map'][()] 
rate_phase_map=batch_data['final_rate_phase_map'][()]

idxs=np.array(rate_score_map.values())>0.5
print '%d/100 with score higher than 0.5'%sum(idxs)

scores=np.array(rate_score_map.values())[:]
spacings=np.array(rate_spacing_map.values())[idxs]
angles=np.array(rate_angle_map.values())[idxs]
phases=[]
num_idxs=np.where(idxs)[0]
for idx in num_idxs:
  phases.append(rate_phase_map.values()[idx])
phases=np.array(phases)

# compute max_freqs
rates_map_mat=np.array(rates_map.values())[idxs]
rates_map_mat=np.swapaxes(rates_map_mat,0,2)
rates_map_mat=np.swapaxes(rates_map_mat,0,1)
rates_dfts,freqs,allfreqs=gl.dft2d_num(rates_map_mat,p.L,p.nx)
profiles=gl.dft2d_profiles(rates_dfts)
max_freqs=freqs[np.argmax(profiles,axis=1)]
  

if len(rates_map)>0:
  all_scores=[]
  for idx,par_values in enumerate(batch.all_par_values):
    all_scores.append(rate_score_map[par_values])

  sorted_scores, sorted_par_values = zip(*sorted(zip(all_scores, batch.all_par_values),reverse=True))
else:
  sorted_par_values=batch.all_par_values





#%%

inputs=GridInputs(batch_default_map,comp_scores=True)

#### EXAMPLE INPUTS

pl.rc('font',size=11)
pp.set_tick_size(3)



pl.figure(figsize=(5.5,2))
input_idxs=[0,5,2,3,7]
idx=0
for idx,input_idx in enumerate(input_idxs):
  pl.subplot(1,len(input_idxs),idx+1,aspect='equal')
  pp.noframe()
  rmap=inputs.inputs_flat[:,input_idx].reshape(inputs.nx,inputs.nx)
  pl.pcolormesh(rmap,vmin=0,rasterized=True)
  
  pl.title('%.1f '%(rmap.max()))

  
  #pp.colorbar(num_int=3)

pl.savefig(figures_path+'/fig9a_maps.eps',bbox_inches='tight',dpi=300)



#%%
pl.figure(figsize=(5.5,2))
idx=0
for idx,input_idx in enumerate(input_idxs):
  pl.subplot(1,len(input_idxs),idx+1,aspect='equal')
  pl.xticks([])
  pl.yticks([])
  pp.plot_matrixDFT(inputs.inputs_flat[:,input_idx].reshape(inputs.nx,inputs.nx),p.L/p.nx,circle_radius=3.,cmap='binary',circle_color=pp.red)
  
pl.savefig(figures_path+'/fig9a_dfts.eps',bbox_inches='tight',dpi=300)

  

#%%
#### EXAMPLE OUTPUTS

pl.rc('font',size=11)
pp.set_tick_size(3)


pl.figure(figsize=(5.5,2))
output_seeds=[87,89,56,6]
idx=0
for idx,output_seed in enumerate(output_seeds):
  pl.subplot(1,len(output_seeds),idx+1,aspect='equal')
  pp.noframe()
  rmap=rates_map[(output_seed,)]
  pl.pcolormesh(rmap,vmin=0,rasterized=True)
  pl.title('%.1f %.2f'%(rmap.max(),rate_score_map[(output_seed,)]))
  
pl.savefig(figures_path+'/fig9b_maps.eps',bbox_inches='tight',dpi=300)

#%%
pl.figure(figsize=(5.5,2))
idx=0
for idx,output_seed in enumerate(output_seeds):
  pl.subplot(1,len(output_seeds),idx+1,aspect='equal')
  pl.xticks([])
  pl.yticks([])
  rmap=rates_map[(output_seed,)]
  pp.plot_matrixDFT(rmap,p.L/p.nx,circle_radius=3.,cmap='binary',circle_color=pp.red,plot_circle=True)
  
  
pl.savefig(figures_path+'/fig9b_dfts.eps',bbox_inches='tight',dpi=300)



#%%

# scores
pl.figure(figsize=(8,1.7))  
pl.subplots_adjust(left=0.1,right=1,top=0.85,bottom=0.3,wspace=0.2)

pl.subplot(1,3,1)
bins,edges=np.histogram(scores,range=[-1.,2],bins=30)
pl.bar(edges[:-1],bins.astype(np.float)/sum(idxs),color='k',width=edges[1]-edges[0],align='center')
pp.custom_axes()
pl.axvline(np.mean(scores),color=pp.red,lw=2)
pl.xlim(-1,2)
pl.xticks([-1,0,1,2])
pl.yticks([0,0.1,0.2])
pl.xlabel('Gridness score')
pl.ylabel('Fraction of cells')
pl.savefig(figures_path+'/fig9c.eps', dpi=300,transparent=True)




pl.figure(figsize=(1.7,1.7))
pl.subplots_adjust(left=0.4,right=0.9,top=0.85,bottom=0.3,wspace=0.2,hspace=0.6)

pl.subplot(1,1,1)
bins,edges=np.histogram(max_freqs,range=[0.,5],bins=10)
pl.bar(edges[:-1],bins.astype(np.float)/sum(idxs),color='k',width=edges[1]-edges[0],align='center')
pp.custom_axes()
pl.yticks([0,.5,1])
pl.xticks([0,1,2,3,4])
pl.xlim(0,4)
pl.xlabel('Frequency [1/m]')
pl.ylabel('Fraction of cells')
pl.savefig(figures_path+'/fig9d.eps', dpi=300,transparent=True)


# phases
pl.figure(figsize=(1.7,1.7))
pl.subplots_adjust(left=0.4,right=0.9,top=0.85,bottom=0.3,wspace=0.2,hspace=0.6)

pl.subplot(1,1,1,aspect='equal')
pl.plot(phases[:,0],phases[:,1],'.k',ms=7)
pp.custom_axes()
pl.xlim(-0.3,0.3)
pl.ylim(-0.3,0.3)
pl.xticks([-0.2,0.,0.2])
pl.yticks([-0.2,0.,0.2])
pl.savefig(figures_path+'/fig9e.eps', dpi=300,transparent=True)

# angles
pl.figure(figsize=(8,1.7))  
pl.subplots_adjust(left=0.1,right=1,top=0.85,bottom=0.3,wspace=0.2)

pl.subplot(1,3,1)
bins,edges=np.histogram(np.array(angles)*180/np.pi,range=[0,60],bins=30)
pl.bar(edges[:-1],bins.astype(np.float)/sum(idxs),width=edges[1]-edges[0],color='k')
pp.custom_axes()
pl.xlim(0,60)
pl.xlabel('Grid Orientation [degrees]')
pl.ylabel('Fraction of cells')
pl.xticks([0,15,30,45,60])
pl.yticks([0,0.1,0.2])
pl.ylim(0,0.2)
pl.savefig(figures_path+'/fig9f.eps', dpi=300,transparent=True)

