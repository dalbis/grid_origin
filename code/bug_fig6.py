#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:40:01 2019

@author: dalbis
"""


import numpy as np
import pylab as pl
import os
from grid_const import ModelType
from grid_functions import map_merge
from grid_params import GridSpikeParams
from grid_batch import GridBatch
from simlib import ensureDir
from grid_walk import GridWalk

figures_path='../figures'
ensureDir(figures_path)


batch_default_map_var_speed=map_merge(
                GridSpikeParams.gau_grid_small_arena_biphasic_neg,
                {
                'a':1.1,
                'variable_speed':True,
                'sim_time':1e6,
                'walk_time':1e4
                
                })
  
batch_override_map = {'walk_seed': np.arange(40)}

batch_var=GridBatch(ModelType.MODEL_SPIKING,batch_default_map_var_speed,batch_override_map)
batch_var.post_init()

if not os.path.exists(batch_var.batch_data_path):
  print 'Running simulations with variable speed'
  batch_var.run()
  batch_var.post_run()
  
  
batch_data=np.load(batch_var.batch_data_path)
all_evo_weight_scores=batch_data['evo_weight_scores_map'][()].values()
all_evo_weight_scores_mat=np.array(all_evo_weight_scores)  
 
#%%

pl.figure()
pl.plot(all_evo_weight_scores_mat.T) 
#%% 
  
  
pl.figure()
idx=1
for h in batch_var.hashes[0:5]:
  fname='/home/dalbis/code/modeldb_dalbis_2018/results/grid_spikes/%s_data.npz'%h
  #print time.ctime(os.path.getmtime(fname) )
  data=np.load(fname)


  J_vect=data['J_vect']
  
  #pl.subplot(1,5,idx)  
  #pl.pcolormesh(J_vect)
  pl.plot(J_vect[0,:])
  idx+=1

################
  
#%%

walk=GridWalk(map_merge(batch_default_map_var_speed,
                              {
                             'arena_shape':'square',  
                             'virtual_bound_ratio':1.0,
                             'bounce_theta_sigma':0.0,
                             'position_dt':1/200.  }) ) 
walk.plot()