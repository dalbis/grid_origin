# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:08:32 2015

@author: dalbis
"""



import numpy as np
from simlib import print_progress,run_from_ipython
import os
from copy import deepcopy
from numpy import diag
from grid_corr_space import GridCorrSpace
from grid_rate import GridRate 
from grid_params import GridRateParams
from grid_inputs import GridInputs


class GridRateAvg(GridRate):
  """
  Rate based
  """
  
  results_path='../results/grid_rate_avg'
  
  def __init__(self,paramMap):
    
    super(GridRateAvg, self).__init__(paramMap)    
    self.header_str="""
    =================================================================
                       AVERAGE RATE SIMULATION                   
    ================================================================="""
      
      
    self.paramsPath=os.path.join(GridRateAvg.results_path,self.hash_id+'_log.txt')
    self.dataPath=os.path.join(GridRateAvg.results_path,self.hash_id+'_data.npz')
    self.figurePath=os.path.join(GridRateAvg.results_path,self.hash_id+'_fig.png')
    
    
    if os.path.exists(self.dataPath):
      #print 'Data hash %s already present'%self.dataPath
      self.do_run=False
    else:
      self.do_run=True
    
    
    
  def run(self,do_print=False) :
     
    # learning constants (k notation)
    self.k1=self.B
    self.k2=self.gamma*self.input_mean
    self.k3=self.a
    
    # load correlation matrix
    corr=GridCorrSpace(self.initParamMap,do_print=do_print,
                           force_gen_inputs=self.force_gen_inputs,force_gen_corr=self.force_gen_corr)
    self.C=corr.CC_teo


    self.m=diag(np.ones(self.N)) 
    self.M=np.ones((self.N,self.N))
    
    # dynamical system matrix
    self.A=self.C-self.m*self.k3-self.M*self.k2
    
    
    self.snap_idx=0
      
    self.r_out=self.r0
    self.rout_av=self.r0
    
    self.J=deepcopy(self.J0)
    self.dJ=np.zeros_like(self.J)

    self.J_vect=np.zeros((self.N,self.num_snaps))
    self.dJ_vect=np.zeros((self.N,self.num_snaps))
    self.r_out_vect=np.zeros(self.num_snaps)

    #### --------------- code to correct boundary effects code --------------

    if self.correct_border_effects is True:
      centers=GridInputs(self.initParamMap).centers
      cx=centers[:,0]
      cy=centers[:,1]
  
      # distance to the center from x and y border
      dbx=self.L/2-np.array([np.abs(cx-self.L/2),np.abs(cx+self.L/2)]).min(axis=0)
      dby=self.L/2-np.array([np.abs(cy-self.L/2),np.abs(cy+self.L/2)]).min(axis=0)
        
      
      border_edge_smooth_fun = lambda x : np.cos(2*np.pi*x/(4*(self.L/2-self.L/2*self.border_size_perc)))
      border_edge_sharp_fun = lambda x : 0.
      
      if self.border_edge_type is 'smooth':
        border_edge_fun=border_edge_smooth_fun
      else:
        border_edge_fun=border_edge_sharp_fun
  
      border_fun = np.vectorize(lambda x : 1. if x<=self.L/2.*self.border_size_perc else  border_edge_fun(x-self.L/2.*self.border_size_perc))
  
  
      self.border_envelope=border_fun(dbx).reshape(self.n,self.n)*border_fun(dby).reshape(self.n,self.n)
  
      self.border_envelope_flat=self.border_envelope.reshape(self.n**2) 
    
    #---------------------------------------------
    
    t=0.
    
    # run the simulation
    for step_idx in xrange(self.num_sim_steps):
         
      t+=self.dt
      
      # save variables
      if np.remainder(step_idx,self.delta_snap)==0:  
        print_progress(self.snap_idx,self.num_snaps,self.startClock)
        self.J_vect[:,self.snap_idx]=self.J
        self.dJ_vect[:,self.snap_idx]=self.dJ        
        self.snap_idx+=1        

      self.dJ=(np.dot(self.A,self.J)+self.k1)
      
      if self.correct_border_effects is True:
        self.J+=self.dt*self.eta*self.dJ*self.border_envelope_flat
      else:
        self.J+=self.dt*self.eta*self.dJ
        
      self.J=np.clip(self.J,0,self.up_bound)           


    
  def plot_weights(self,snap_idx=-1):
    import pylab as pl
    from plotlib import noframe,colorbar,plot_weight_dist,custom_axes
    from numpy import arange
    snap_times=arange(self.num_snaps)*self.delta_snap*self.dt
    
    final_J_mat=self.J_vect[:,snap_idx].reshape(self.n,self.n)           
    pl.figure()
    pl.subplot(211,aspect='equal')
    pl.pcolormesh(final_J_mat)
    noframe()    
    colorbar()
    
    pl.subplot(212)
    custom_axes()
    plot_weight_dist(snap_times,self.J_vect,alpha=0.2)    


if __name__ == '__main__':
  par_map= GridRateParams.gau_grid_small_arena_biphasic_neg                      
  sim=GridRateAvg(par_map)      
  sim.post_init()

  if sim.do_run and not run_from_ipython():
    sim.run()
    sim.post_run()
  elif run_from_ipython():
    sim.plot_eigs()

