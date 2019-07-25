# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:25:22 2017

@author: dalbis
"""


import numpy as np
import pylab as pl
import plotlib as pp
import os
from grid_const import ModelType
from grid_functions import map_merge
from grid_params import GridSpikeParams
from grid_batch import GridBatch
from simlib import ensureDir

figures_path='../figures'
ensureDir(figures_path)
  
#%%

######################## MULTILPLE LEARNING RATES ###################################


learn_rates=[2e-5,3e-5,5e-5,1e-4]
batch_default_map=map_merge(
                GridSpikeParams.gau_grid_small_arena_biphasic_neg,
                {
                'a':1.1                
                })

                
# time vector
num_sim_steps = int(batch_default_map['sim_time']/batch_default_map['dt'])
delta_snap = int(num_sim_steps/batch_default_map['num_snaps'])
snap_times=np.arange(batch_default_map['num_snaps'])*delta_snap*batch_default_map['dt']

      
pl.rc('font',size=13)
pp.set_tick_size(4)

fig=pl.figure(figsize=(8,3),facecolor='w')
ax1=pl.subplot(1,2,1)
ax2=pl.subplot(1,2,2)
pl.subplots_adjust(bottom=0.2,wspace=0.4,left=0.1,right=0.97)

colormap = pl.cm.gist_heat_r
pl.sca(ax1)
pl.gca().set_color_cycle([colormap(i) for i in np.linspace(0.2, 1.0, len(learn_rates))])


for eta in reversed(learn_rates):                
  batch_override_map = {'seed':np.arange(40),'eta':(eta,)}
    
  batch=GridBatch(ModelType.MODEL_SPIKING,batch_default_map,batch_override_map)
  batch.post_init()
  if not os.path.exists(batch.batch_data_path):
    
    print 'Running simulations for learning rate %.3e  (for 40 different initial weights)'%eta
    batch.run()
    batch.post_run()

  batch_data=np.load(batch.batch_data_path)
  all_evo_weight_scores=batch_data['evo_weight_scores_map'][()].values()
  all_evo_weight_scores_mat=np.array(all_evo_weight_scores)
  median_scores=np.median(all_evo_weight_scores_mat,axis=0)
  perc_25=np.percentile(all_evo_weight_scores_mat,25,axis=0)
  perc_75=np.percentile(all_evo_weight_scores_mat,75,axis=0)
  pl.sca(ax1)

  pl.plot(snap_times,median_scores,'-',lw=1.5)  
  
pl.sca(ax1)
pp.custom_axes()
pl.gca().set_xscale('log')
pl.xlim(1e5,snap_times.max())
pl.ylim(0,1.5)
pl.xlabel('Time [10^5 s]')
pl.ylabel('Gridness score')
pl.yticks([0,0.5,1.0,1.5])
pl.xticks([1e5,2e5,5e5,1e6],['1','2','5','10'])  

#%%

##### PLOT CONSTANT VS VARIABLE SPEED ======================================================
                
pl.sca(ax2)
batch_override_map = {'seed': np.arange(40)}

# constant speed
batch=GridBatch(ModelType.MODEL_SPIKING,batch_default_map,batch_override_map)
batch.post_init()

if not os.path.exists(batch.batch_data_path):
  print 'Running simulations with constant speed (for 40 different initial weights)'
  batch.run()
  batch.post_run()
  
  
batch_data=np.load(batch.batch_data_path)
all_evo_weight_scores=batch_data['evo_weight_scores_map'][()].values()
all_evo_weight_scores_mat=np.array(all_evo_weight_scores)

pl.plot(snap_times,all_evo_weight_scores_mat.T,'--k')

median_scores=np.median(all_evo_weight_scores_mat,axis=0)

pl.plot(snap_times,median_scores,'-k',lw=1.5)  
#pl.plot(snap_times,all_evo_weight_scores_mat.T)

#%%
batch_default_map_var_speed=map_merge(
                GridSpikeParams.gau_grid_small_arena_biphasic_neg,
                {
                'a':1.1,
                'variable_speed':True,
                })
  
batch_override_map = {'seed': np.arange(40)}
  

# variable speed
batch_var=GridBatch(ModelType.MODEL_SPIKING,batch_default_map_var_speed,batch_override_map)
batch_var.post_init()
if not os.path.exists(batch_var.batch_data_path):
  print 'Running simulations with variable speed (for 40 different initial weights)'
  batch_var.run()
  batch_var.post_run()
batch_data=np.load(batch_var.batch_data_path)
all_evo_weight_scores=batch_data['evo_weight_scores_map'][()].values()
all_evo_weight_scores_mat=np.array(all_evo_weight_scores)
pl.plot(snap_times,all_evo_weight_scores_mat.T,'g')
median_scores=np.median(all_evo_weight_scores_mat,axis=0)


pl.plot(snap_times,median_scores,'-',lw=1.5,color=pp.green)  
#pl.plot(snap_times,all_evo_weight_scores_mat.T)

pl.sca(ax2)
pp.custom_axes()
pl.gca().set_xscale('log')
pl.xlim(1e5,snap_times.max())
pl.ylim(0,1.5)
pl.xlabel('Time [10^5 s]')
pl.ylabel('Gridness score')
pl.yticks([0,0.5,1.0,1.5])
pl.xticks([1e5,2e5,5e5,1e6],['1','2','5','10'])   

fig.savefig(os.path.join(figures_path,'fig6.eps'),dpi=300,transparent=True)      

##%%
#from grid_walk import GridWalk
#walk=GridWalk(map_merge(batch_default_map_var_speed,
#                            {'arena_shape':'square',  
#                             'virtual_bound_ratio':1.0,
#                             'bounce_theta_sigma':0.0,
#                             'position_dt':1/200.  }))
#
#
#fig=pl.figure(figsize=(1.5,1.))
#pl.subplots_adjust(left=0.2,bottom=0.2)
#n,bins,patches=pl.hist(walk.speed_vect,100,color=pp.green,edgecolor=pp.green,normed=1)
#pl.yticks([0,25])
#pl.xticks([.15,.25,.35])
#pp.custom_axes()
#    
#fig.savefig(os.path.join(figures_path,'fig6b_inset.eps'),dpi=300,transparent=True)      

