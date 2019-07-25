# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:04:57 2016

@author: dalbis
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:30:16 2015

@author: dalbis
"""

import sys
from multiprocessing import Pool
from grid_rate import GridRate
from grid_rate_avg import GridRateAvg
from grid_spikes import GridSpikes
import itertools
import socket
import traceback
import os
import numpy as np
import datetime,time
from simlib import gen_hash_id,format_val,logSim,format_elapsed_time,print_progress,ensureDir,ensureParentDir
from grid_const import ModelType
from grid_functions import map_merge

# maximum processes per host
procs_by_host={'compute1':50,
               'compute2':50, 
               'compute3':8,
               'cluster01':20, 
               'cluster02':5,
               'mcclintock':6}


# batch data path

batch_data_folder_map={
    ModelType.MODEL_RATE:'../results/grid_rate_batch',
    ModelType.MODEL_RATE_AVG:'../results/grid_rate_avg_batch',
    ModelType.MODEL_SPIKING:'../results/grid_spiking_batch'
}


def function(sim):

  try:
    if sim.do_run is True:
      sim.post_init(do_print=False)
      sim.run(do_print=False)
      sim.post_run(do_print=False)  
  
  except Exception:
    print
    print 'Exception in running %s'%sim.dataPath
    traceback.print_exc()
  


class GridBatch(object): 
  
  
  def __init__(self,model_type,batch_default_map,batch_override_map,force=False):
    self.model_type=model_type
    self.batch_default_map=batch_default_map
    self.batch_override_map=batch_override_map
    self.force=force
    
    self.startTimeStr=''
    self.endTimeStr=''
    self.elapsedTime=0
    
  def post_init(self):

    ##############################################################################
    ###### CREATE POOL
    ##############################################################################

    self.batch_data_folder=batch_data_folder_map[self.model_type]      
    ensureDir(self.batch_data_folder)

    # create pool  
    self.host=socket.gethostname()  
    if self.host in procs_by_host.keys():
      self.num_procs=procs_by_host[self.host]
    else:
      self.num_procs=7
      
    self.pool=Pool(processes=self.num_procs)
    self.sims=[]
    self.hashes=[]
  
    self.all_par_values=sorted(itertools.product(*self.batch_override_map.values()))  
    self.batch_override_str=' '.join([ '%s (%s-%s)'%(key,format_val(min(values)),                                            
format_val(max(values))) for key,values in self.batch_override_map.items()])
    
    # loop over all different paramater values
    for par_values in self.all_par_values:
  
      override_param_map={k:v for (k,v) in zip(self.batch_override_map.keys(),par_values)} 
      
      parMap=map_merge(self.batch_default_map,override_param_map)
      
      if self.model_type == ModelType.MODEL_RATE:
        self.sim_class=GridRate
      elif self.model_type == ModelType.MODEL_RATE_AVG:
        self.sim_class=GridRateAvg
      elif self.model_type == ModelType.MODEL_SPIKING:
        self.sim_class=GridSpikes
     
      sim=self.sim_class(parMap)    
      #print sim.hash_id+' Run: %s'%sim.do_run
      
      if self.force:
        sim.force_gen_inputs=True
        sim.force_gen_corr=True
        sim.do_run=True
        
      if sim.do_run is True:
        self.sims.append(sim)
        
      self.hashes.append(sim.hash_id)
   
    
    # generate batch hash
    self.batch_hash=gen_hash_id('_'.join(self.hashes))
    self.batch_data_path=os.path.join(self.batch_data_folder,'%s_data.npz'%self.batch_hash)
    self.batch_params_path=os.path.join(self.batch_data_folder,'%s_params.txt'%self.batch_hash)    
    
    
    
    self.batch_summary_str=\
    "\n\nBATCH HASH: %s\n\nBATCH PARAMS = %s\n\n"%\
    (self.batch_hash,
     self.batch_override_str
     )
        
    print self.batch_summary_str
    
    self.toSaveMap={'hashes':self.hashes,
                    'batch_override_map':self.batch_override_map,
                    'batch_default_map':self.batch_default_map
                    }
    
    if os.path.exists(self.batch_data_path) and not self.force:
      return False
    else:
      print '\n\n*** BATCH DATA NOT PRESENT!! ***\n\n' 
      print self.batch_data_path
      print '%d/%d simulations to be run'%(len(self.sims),len(self.all_par_values))
      return True
     
     
    
     
  def run(self):
    
    
    ##############################################################################
    ###### RUN POOL
    ##############################################################################
    
    startTime=time.time()
    startTimeDate=datetime.datetime.fromtimestamp(time.time())
    self.startTimeStr=startTimeDate.strftime('%Y-%m-%d %H:%M:%S')
    
    print 'BATCH MODE: Starting %d/%d processes on %s'%(len(self.sims),self.num_procs,self.host)
    
    for sim in self.sims:  
      self.pool.apply_async(function, args=(sim,))
  
    self.pool.close()
    self.pool.join()      
  
  
    # logging simulation end
    endTime=datetime.datetime.fromtimestamp(time.time())
    self.endTimeStr=endTime.strftime('%Y-%m-%d %H:%M:%S')
    self.elapsedTime =time.time()-startTime
  
    print 'Batch simulation ends: %s'%self.endTimeStr
    print 'Elapsed time: %s\n' %format_elapsed_time(self.elapsedTime)
      
      
  def post_run(self):
      
      
    #############################################################################
    ##### MERGE DATA
    #############################################################################
  
    print
    print 'SIMULATIONS COMPLETED'
    print
    print 'Merging data...'
    sys.stdout.flush()
     
  
    initial_weights_map={}
     
    final_weights_map={}
    final_weight_score_map={}
    final_weight_angle_map={}
    final_weight_spacing_map={}
    final_weight_phase_map={}
    final_weight_cx_map={}
    evo_weight_scores_map={}
    
    final_rates_map={}
    final_rate_score_map={}
    final_rate_angle_map={}
    final_rate_spacing_map={}
    final_rate_phase_map={}
    final_rate_cx_map={}
    
    evo_weight_profiles_map={}
    
    start_clock=time.time()
    
    # load/compute data to show for each combination of parameter_values
    idx=-1
    for chash,par_values in zip(self.hashes,self.all_par_values):
      idx+=1
      print_progress(idx,len(self.all_par_values),start_clock=start_clock)
      sys.stdout.flush()
  
      dataPath=os.path.join(self.sim_class.results_path,'%s_data.npz'%chash)
      
      try:
        data=np.load(dataPath,mmap_mode='r')
      except Exception:
        print 'This file is corrupted: %s'%dataPath
          
  
      initial_weights_map[par_values]=data['J0']
      final_weights_map[par_values]=data['final_weights']
      final_weight_score_map[par_values]=data['final_weight_score']
      final_weight_angle_map[par_values]=data['final_weight_angle']
      final_weight_spacing_map[par_values]=data['final_weight_spacing']
      final_weight_phase_map[par_values]=data['final_weight_phase']
      final_weight_cx_map[par_values]=data['final_weight_cx']
      if 'scores' in data.keys():
        evo_weight_scores_map[par_values]=data['scores']
      
      final_rates_map[par_values]=data['final_rates']    
      final_rate_score_map[par_values]=data['final_rate_score']
      final_rate_angle_map[par_values]=data['final_rate_angle']
      final_rate_spacing_map[par_values]=data['final_rate_spacing']
      final_rate_phase_map[par_values]=data['final_rate_phase']
      final_rate_cx_map[par_values]=data['final_rate_cx']



      # fourier profiles over time
      import gridlib as gl
      L=data['paramMap'][()]['L']
      n=data['paramMap'][()]['n']
      num_snaps=self.batch_default_map['num_snaps']
      J_mat=data['J_vect'].reshape(n,n,num_snaps)  
      weights_dft,weights_freqs,weigths_allfreqs=gl.dft2d_num(J_mat,L,n)
      weights_dft_profiles=gl.dft2d_profiles(weights_dft)   
      evo_weight_profiles_map[par_values]=weights_dft_profiles
      

    
  
        
  
    
    mergedDataMap={
                   'initial_weights_map': initial_weights_map,
                   'final_weights_map': final_weights_map,
                   'final_weight_score_map':final_weight_score_map,
                   'final_weight_angle_map':final_weight_angle_map,
                   'final_weight_spacing_map':final_weight_spacing_map,
                   'final_weight_phase_map':final_weight_phase_map,
                   'final_weight_cx_map':final_weight_cx_map,
                   'evo_weight_scores_map':evo_weight_scores_map,
                   
                   'final_rates_map':final_rates_map,
                   'final_rate_score_map':final_rate_score_map,
                   'final_rate_angle_map':final_rate_angle_map,
                   'final_rate_spacing_map':final_rate_spacing_map,
                   'final_rate_phase_map':final_rate_phase_map,
                   'final_rate_cx_map':final_rate_cx_map,
                   
                   'evo_weight_profiles_map':evo_weight_profiles_map,
                   'weights_freqs':weights_freqs
                   }
    
    self.toSaveMap=map_merge(self.toSaveMap,mergedDataMap)
        
    # save      
    ensureParentDir(self.batch_data_path)
    logSim(self.batch_hash,self.batch_override_str,self.startTimeStr,self.endTimeStr,self.elapsedTime,self.batch_default_map,self.batch_params_path,doPrint=False)
    
    print
    print 'BATCH HASH: %s'%self.batch_hash
    np.savez(self.batch_data_path,**self.toSaveMap)
    print 
    print 'Batch data saved in: %s\n'%self.batch_data_path
    print  


    
    
  
