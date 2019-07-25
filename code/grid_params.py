# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 11:44:32 2016

@author: dalbis
"""


from grid_const import InputType,DistType,FilterType
from grid_functions import map_merge

def_spikes={
        
  # general parameters
  'n':30,
  'dt':0.001,
  'num_snaps':200,
  'eta':2e-5,
  'sim_time':1e6,
  'transient_time':2.,
  'seed':3,

  # walk parameters
  'L':1.,
  'nx':200,
  'speed':0.25,
  'theta_sigma' : 0.7,
  'walk_time':1e6, #5000
  'walk_seed':0,    
  'periodic_walk':True,
  'bounce':False,
  'bounce_theta_sigma':0.0,
  'variable_speed':False,
  'speed_theta':10.,
  'speed_sigma':0.1, 
                
  # input parameters
  'periodic_inputs':True,
  'add_boundary_input':False,

  
  'inputs_type':InputType.INPUT_GAU_GRID,
  'centers_std':0.02,
  'inputs_seed':0,
  'outside_ratio':1.0,
  'virtual_bound_ratio':1.0,
  
  
  # eigenvalue threshold and desired average weight    
  'J_av_target':0.05,
  
  'up_bound':1.,  
  'J0_std':1e-4,
  'r0':10.,
  
  # STDP
  'Aplus' : 10.,
  'Aminus' : 10.,
  'tau_plus' : 0.05,
  'tau_minus' : 0.05,
  
  'compute_scores':True,
}   

def_spikes_sigma_large=map_merge(def_spikes,
                                {
                                  'sigma':0.0625,
                                  'input_mean':0.4,
                                  'a':1.0
                                })

def_spikes_sigma_small=map_merge(def_spikes,
                                {
                                  'sigma':0.04,
                                  'input_mean':0.2,
                                  'a':5.0
                                })         
      

#### RATE      

def_rate ={

  # general parameters
  'dt':0.001,
  'sim_time':1e6,
  'eta':5e-5,
  'seed':0,
  'num_snaps':200,

  'nx':200,
  'periodic_walk':True,
  'speed':0.25,
  'bounce':False,
  'theta_sigma': 0.7,
  'walk_time':1e6,
  'walk_seed':0,
  'variable_speed':False,
  'speed_theta':10.,
  'speed_sigma':0.1,

  'periodic_inputs':True,
  'outside_ratio':1.0,
  
  'J_av_target':0.05,

  'up_bound':1.0,
  'J0_std':1e-3,
  'r0':4.,
  'J0_dist':DistType.J0_NORM,
  'J0_mean_factor':1.,
  
  'filter_type': FilterType.FILTER_INPUT,
  'add_boundary_input':False,
  'clip_weights':True,
  'clip_out_rate':False,
  'compute_scores':True
  }
  

# Gau grid small arena
def_gau_grid_small_arena_sigma_large=map_merge(def_rate,
  {
   'L':1.0,
   'n':30,
   'input_mean':0.4,
   'sigma':0.0625,
   'inputs_type':InputType.INPUT_GAU_GRID,
   'a':1.0,
  })                                   
   

def_gau_grid_small_arena_sigma_small=map_merge(def_rate,
  {
   'L':1.0,  
   'n':30,
   'input_mean':0.2,
   'sigma':0.04,
   'inputs_type':InputType.INPUT_GAU_GRID,
   'a':5.0,
  })      
  

# Gau mix small arena   
def_gaumix_small_arena_sigma_large=map_merge(def_rate,
  {
  'L':1.0,
  'n':60,
  'input_mean':0.8,
  'sigma':0.0625,
  'inputs_type':InputType.INPUT_GAU_MIX_POS,
  'num_gau_mix':10,
  'inputs_seed':89,
  'a':2.5,
  'J_av_target':0.02,

  })    

def_gaumix_small_arena_sigma_small=map_merge(def_rate,
  {
  'L':1.0,
  'n':60,
  'input_mean':0.4,
  'sigma':0.04,
  'inputs_type':InputType.INPUT_GAU_MIX_POS,
  'num_gau_mix':10,
  'inputs_seed':89,
  'a':2.5,
  'J_av_target':0.02,

  })    
    

### Large arena
    
def_gaugrid_large_arena_sigma_large=map_merge(def_rate,
 {
    'L':2.0,
    'n':60,
    'input_mean':0.1,
    'sigma':0.0625,
    'inputs_type':InputType.INPUT_GAU_GRID,
    'a':0.25,

  })
  
def_gaugrid_large_arena_sigma_small=map_merge(def_rate,
 {
    'L':2.0,
    'n':60,
    'input_mean':0.105,
    'sigma':0.04,
    'inputs_type':InputType.INPUT_GAU_GRID,
    'a':5.5,
  })
    
  
### Filters
 

filter_biphasic_neg={
  'tau1':0.1,
  'tau2':0.16,
  'tau3':0.7,

  'mu1':1.,
  'mu2':-1.06,
  'mu3':0.0,

  'gamma':0.0,  
  }


filter_biphasic_pos={
  
  'tau1':0.05,
  'tau2':0.2,
  'tau3':0.7,

  'mu1':1.,
  'mu2':-0.97,
  'mu3':0.0,

  'gamma':0.03,  
  }
    
    
class GridSpikeParams:
  gau_grid_small_arena_biphasic_neg=map_merge(def_spikes_sigma_large,filter_biphasic_neg)
  gau_grid_small_arena_biphasic_pos=map_merge(def_spikes_sigma_small,filter_biphasic_pos)    
  

class GridRateParams:  

  # gaugrid small arena  
  gau_grid_small_arena_biphasic_neg = map_merge(def_gau_grid_small_arena_sigma_large,filter_biphasic_neg)
  gau_grid_small_arena_biphasic_pos = map_merge(def_gau_grid_small_arena_sigma_small,filter_biphasic_pos)
  
  # gaugrid large arena
  gau_grid_large_arena_biphasic_neg = map_merge(def_gaugrid_large_arena_sigma_large,filter_biphasic_neg)
  gau_grid_large_arena_biphasic_pos = map_merge(def_gaugrid_large_arena_sigma_small,filter_biphasic_pos)
  
  # gaumix small arena
  gau_mix_small_arena_biphasic_neg = map_merge(def_gaumix_small_arena_sigma_large,filter_biphasic_neg)
  gau_mix_small_arena_biphasic_pos = map_merge(def_gaumix_small_arena_sigma_small,filter_biphasic_pos)

  

                                             














