# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:08:32 2015

@author: dalbis
"""



from gridlib import compute_scores_evo
import numpy as np
from numpy.random import randn,seed
from simlib import print_progress,params_to_str,logSim,format_elapsed_time
from simlib import gen_string_id,gen_hash_id,run_from_ipython,ensureParentDir
from time import clock
import datetime,time
import os
from grid_inputs import GridInputs
from grid_walk import GridWalk
from grid_functions import compute_teo_eigs,filter_r_vect,get_step_response,map_merge,K_outeq_ft_k
from copy import deepcopy
from numpy.random import exponential,permutation
from grid_const import DistType,InputType,FilterType
from gridlib import get_grid_params
from grid_params import GridRateParams


class GridRate(object):
  """
  Rate based
  """
  
  results_path='../results/grid_rate'
  
  key_params_filter_input=['sim_time','eta','dt','a','J_av_target','gamma','tau1','tau2','tau3','mu1','mu2','mu3',
             'up_bound','seed','r0','clip_out_rate','J0_std','add_boundary_input','J0_dist','sigmoid_out_rate','r_out_max']

  key_params_filter_output=['sim_time','eta','dt','a','J_av_target','gamma','tau_in','tau_out','mu_out',
             'up_bound','seed','r0','clip_out_rate','J0_std','add_boundary_input','J0_dist','sigmoid_out_rate','r_out_max']
  
  def __init__(self,paramMap,scale_a_by_density=False,force=False):

    
    self.header_str="""
    =================================================================
                     GRID DETAILED RATE SIMULATION                   
    ================================================================="""
    
    
    self.force_gen_inputs=False
    self.force_gen_corr=False    
    self.force=force

    # general parameters
    self.dt=None
    self.sim_time=None
    self.eta=None
    self.seed=None
    self.num_snaps=None

    # input parameters
    self.n=None
    self.input_mean=None
    self.sigma=None
    self.inputs_type=None
    self.periodic_inputs=None
    self.num_gau_mix=None
    self.centers_std=None
    self.inputs_seed=None
    self.tap_inputs=None
    self.norm_bound_add=None
    self.norm_bound_mul=None
    
    # walk parameters
    self.L=None
    self.nx=None
    self.periodic_walk=None
    self.speed=None
    self.bounce=None
    self.theta_sigma = None
    self.position_dt=None
    self.walk_time=None
    self.walk_seed=None
    self.variable_speed=None
    
    
    self.correct_border_effects=None
    
    # filter parameters
    
    self.filter_type=FilterType.FILTER_INPUT

    if 'filter_type' not in paramMap.keys() or paramMap['filter_type']==FilterType.FILTER_INPUT:
      self.tau1=None
      self.tau2=None
      self.tau3=None
      self.mu1=None
      self.mu2=None
      self.mu3=None
    else:
      self.tau_in=None
      self.tau_out=None
      self.mu_out=None

    # plasticiy params
    self.a=None
    self.J_av_target=None
    self.gamma=None    
    self.up_bound=None
    self.J0_std=None
    self.r0=None
    self.J0_dist=None
    self.J0_mean_factor=None     
    self.base_n=None          # base number of neurons to scale a accordingly
    
    # flags
    self.add_boundary_input=None
    self.clip_weights=None
    self.clip_out_rate=None
    self.compute_scores=None
    self.scale_a_by_n=None

    # set parameter values from input map
    for param,value in paramMap.items():
      setattr(self,param,value)

    if self.scale_a_by_n is True:
      self.a=self.a*self.n**2/self.base_n**2
      
    # parameters we never change
    self.plastic=True
    self.transient_time=2.
    self.arena_shape='square'
    self.virtual_bound_ratio=1.0
    self.bounce_theta_sigma=0.0
    self.debug_vars=False
    self.sigmoid_out_rate=False
    self.r_out_max=5.
    self.position_dt=self.L/self.nx    

      
    if self.periodic_inputs is True:
      assert(self.add_boundary_input==False)
      
    if (self.mu1+self.mu2+self.mu3)<0.:
      assert(self.gamma==0.0)
    else:
      assert(self.gamma>0.0)

      
    # init parameters map
    self.initParamMap = {
                'arena_shape':self.arena_shape,'L':self.L,'n':self.n,'nx':self.nx,                
                'sigma':self.sigma,'input_mean':self.input_mean,
                'speed':self.speed,'theta_sigma':self.theta_sigma,
                'seed':self.seed,'sim_time':self.sim_time,
                'dt':self.dt,'position_dt':self.position_dt,
                'num_snaps':self.num_snaps,
                'eta':self.eta,'a':self.a,'gamma':self.gamma,
                'J_av_target':self.J_av_target,
                'J0_std':self.J0_std,'up_bound':self.up_bound,'r0':self.r0,
                'J0_mean_factor':self.J0_mean_factor,
                'clip_weights':self.clip_weights,
                'periodic_inputs':self.periodic_inputs,'outside_ratio':self.outside_ratio,
                'clip_out_rate':self.clip_out_rate,
                'inputs_type':self.inputs_type,'num_gau_mix':self.num_gau_mix,
                'inputs_seed':self.inputs_seed,
                'walk_seed':self.walk_seed,'walk_time':self.walk_time,
                'periodic_walk':self.periodic_walk,
                'bounce':self.bounce,'bounce_theta_sigma':self.bounce_theta_sigma,
                'virtual_bound_ratio':self.virtual_bound_ratio,
                'compute_scores':self.compute_scores,
                'add_boundary_input':self.add_boundary_input,                
                'J0_dist':self.J0_dist,
                'sigmoid_out_rate':self.sigmoid_out_rate,
                'r_out_max':self.r_out_max,
                'centers_std':self.centers_std
                }
    if self.filter_type == FilterType.FILTER_INPUT:
      self.initParamMap=map_merge(self.initParamMap,{'tau1':self.tau1,'tau2':self.tau2,'tau3':self.tau3,
                                   'mu1':self.mu1,'mu2':self.mu2,'mu3':self.mu3})
    else:
      self.initParamMap=map_merge(self.initParamMap,{'tau_in':self.tau_in,'tau_out':self.tau_out,
                                   'mu_out':self.mu_out})
              
              
    if self.variable_speed is True:
       self.initParamMap=map_merge(self.initParamMap,
               {
                'variable_speed':self.variable_speed,
                'speed_theta':self.speed_theta,
                'speed_sigma':self.speed_sigma
               })
               
               
    if self.correct_border_effects is True:
       self.initParamMap=map_merge(self.initParamMap,
               {
                'correct_border_effects':self.correct_border_effects,
                'border_edge_type':self.border_edge_type,
                'border_size_perc':self.border_size_perc
               })
               

    if self.tap_inputs is True:
       self.initParamMap=map_merge(self.initParamMap,
               {
                'tap_inputs':self.tap_inputs,
                'tap_border_type':self.tap_border_type,
                'tap_border_size':self.tap_border_size
               })
               
    if self.norm_bound_add is True:
       self.initParamMap=map_merge(self.initParamMap,
               {
                'norm_bound_add':self.norm_bound_add
               })

    if self.norm_bound_mul is True:
       self.initParamMap=map_merge(self.initParamMap,
               {
                'norm_bound_mul':self.norm_bound_mul
               })
                               
    # human-readable parameter strings (just for printing)            
    key_params=GridRate.key_params_filter_input if self.filter_type == FilterType.FILTER_INPUT \
                                                    else GridRate.key_params_filter_output
                                              
    self.key_params_str=params_to_str(self.initParamMap,keyParams=key_params,compact=True)            
    self.input_params_str=params_to_str(self.initParamMap,keyParams=GridInputs.get_key_params(self.initParamMap),compact=True)
    self.walk_params_str=params_to_str(self.initParamMap,keyParams=GridWalk.get_key_params(paramMap),compact=True)
    
    # generate id and paths
    self.str_id=gen_string_id(self.initParamMap)
    self.hash_id=gen_hash_id(self.str_id)
   
    self.paramsPath=os.path.join(GridRate.results_path,self.hash_id+'_log.txt')
    self.dataPath=os.path.join(GridRate.results_path,self.hash_id+'_data.npz')
    self.figurePath=os.path.join(GridRate.results_path,self.hash_id+'_fig.png')
    
    
    if os.path.exists(self.dataPath):
      print 'Data hash %s already present'%self.hash_id
      self.do_run=False
    else:
      self.do_run=True
      
    
  def __getstate__(self):
    return self.__dict__

  def remove_data(self):
    os.remove(self.dataPath)
    os.remove(self.paramsPath)

  def post_init(self,do_print=False):
        
    if not self.force and not self.force_gen_inputs and os.path.exists(self.dataPath):
      if do_print:
        print 'Data hash %s already present'%self.hash_id
      return False
    
    seed(self.seed)


    self.sigmas_per_dx=self.sigma*self.n/self.L
      
    
    self.num_sim_steps = int(self.sim_time/self.dt)
    self.position_dt_scale=int(self.position_dt/self.dt)
      
    self.startClock=clock()
    self.startTime=datetime.datetime.fromtimestamp(time.time())
    self.startTimeStr=self.startTime.strftime('%Y-%m-%d %H:%M:%S')
    

    if self.filter_type==FilterType.FILTER_INPUT:
      # inverse filter time constants
      self.b1=1/self.tau1
      self.b2=1/self.tau2
      self.b3=1/self.tau3
       
      # integral of the filter
      self.K_int=self.mu1+self.mu2+self.mu3

    else:
      self.b_in=1./self.tau_in
      self.b_out=1./self.tau_out
    
      self.K_int = np.real(K_outeq_ft_k(self.b_in,self.b_out,self.mu_out,0.))
      
    # total number of neurons and density
    self.N=self.n**2    
    self.rho=self.N/self.L**2

    # mean input and mean correlation 
    self.C_av=self.input_mean**2*self.K_int

    # compute normalization time constant
    self.tau_av=1./(self.eta*(self.a-self.N*(self.C_av-self.input_mean*self.gamma)))

    # computa B,alpha,beta from a, J_av_target    
    self.B=self.J_av_target/(self.eta*self.tau_av)   
    self.alpha=self.a/self.input_mean
    self.beta=self.B/self.input_mean-self.r0

    self.J_av_star=self.B*self.eta*self.tau_av

    
    # derived quantities
    self.k1=self.input_mean*(self.r0+self.beta)
    self.k2=self.input_mean*self.gamma
    self.k3=self.input_mean*self.alpha
    
  
    # load input data
    self.inputs=GridInputs(self.initParamMap,do_print=do_print,force_gen=self.force_gen_inputs)    
    self.inputs_path=self.inputs.dataPath
    self.inputs_flat=self.inputs.inputs_flat
    
    if self.inputs_type==InputType.INPUT_GAU_GRID:
      self.amp=self.inputs.amp
    else:
      self.amp=np.NaN
      
    # compute eigenvalues    
    self.freqs,self.raw_eigs=compute_teo_eigs(self.inputs,self.__dict__,teo_input_pw=False)

    self.raw_eigs[0]=self.raw_eigs[0]-self.N*self.gamma*self.input_mean
    self.max_eig_idx=self.raw_eigs.argmax()
    self.max_freq=self.freqs[self.max_eig_idx]

    self.eigs_lf_diff=(self.raw_eigs[self.max_eig_idx]-self.raw_eigs[self.max_eig_idx-1])
    self.eigs_hf_diff=(self.raw_eigs[self.max_eig_idx]-self.raw_eigs[self.max_eig_idx+1])


    self.eigs=self.raw_eigs-self.a
    
    self.max_eig=self.eigs.max()
    self.eig0=self.eigs[0]


    self.tau_str=1./(self.eta*self.max_eig)
    
       
    # load walk data
    self.walk=GridWalk(self.__dict__,do_print=do_print)
    self.walk_path=self.walk.dataPath
    self.pos=self.walk.pos
    self.pidx_vect=self.walk.pidx_vect
    self.nx=self.walk.nx
    self.walk_steps=self.walk.walk_steps
    

    # output rate normalization
    self.r_out_star=(self.input_mean*self.K_int-self.gamma)*self.J_av_star*self.N+self.r0


    # compute boundary input
    if self.add_boundary_input is True:
      self.boundary_input_flat=self.r_out_star-self.get_estimated_output(np.ones(self.N)*self.J_av_star,0)
    else:
      self.boundary_input_flat=np.zeros(self.nx**2)
      
    # initial weights    
    self.J0_mean=self.J_av_star*self.J0_mean_factor 
    
    # exponential distribution
    if self.J0_dist == DistType.J0_EXP:
      self.J0=exponential(self.J0_mean,self.N)
      
    # normal distribution  
    elif self.J0_dist == DistType.J0_NORM:
      self.J0=np.ones(self.N)*self.J0_mean+randn(self.N)*self.J0_std
      self.J0=self.J0/np.mean(self.J0)*self.J0_mean
    
    # exponential distribution with half of the weights set to zero      
    elif self.J0_dist == DistType.J0_HALF_EXP:          
      self.J0=np.zeros(self.N)
      non_zero_idxs=permutation(self.N)[0:self.N/2]
      self.J0[non_zero_idxs]=exponential(self.J_av_star*2,self.N/2)  
      self.J0=self.J0/self.J0.mean()*self.J_av_star


    np.clip(self.J0,0,self.up_bound,out=self.J0)   
       

    self.delta_snap = int(np.floor(float(self.num_sim_steps)/(self.num_snaps)))
    assert(self.delta_snap>0)    

    self.derived_param_str = 'amp=%.1f max_eig=%.2f max_freq=%.1f tau_av=%.1e  tau_str=%1.e\
    eigs_lf_diff=%.3f eigs_hf_diff=%.3f'\
    %(self.amp,self.max_eig,self.max_freq,self.tau_av,self.tau_str,self.eigs_lf_diff,self.eigs_hf_diff)

    self.summary_str=  """
    
HASH: %s

KEY PARAMS: %s

INPUT PARAMS: %s

WALK PARAMS: %s

DERIVED PARAMS: %s
      
      """%(self.hash_id,self.key_params_str,self.input_params_str,self.walk_params_str,self.derived_param_str)            

      
    # derived parameters map
    self.derivedParamMap = {
                'hash_id':self.hash_id,
                'N':self.N,'amp':self.amp,
                'rho':self.rho,'sigmas_per_dx':self.sigmas_per_dx,
                'num_sim_steps':self.num_sim_steps,'delta_snap':self.delta_snap,
                'k1':self.k1,'k2':self.k2,'k3':self.k3,
                'K_int':self.K_int,
                'B':self.B,
                'alpha':self.alpha,'beta':self.beta,
                'C_av':self.C_av,
                'J_av_star':self.J_av_star,
                'tau_av':self.tau_av,'r_out_star':self.r_out_star,
                'J0_mean':self.J0_mean,
                'max_eig':self.max_eig,'max_freq':self.max_freq,
                'tau_str':self.tau_str,
                'eig0':self.eig0,
                'eigs_lf_diff':self.eigs_lf_diff,
                'eigs_hf_diff':self.eigs_hf_diff,
                'summary_str':self.summary_str
                }

    if self.filter_type==FilterType.FILTER_INPUT:
      self.derivedParamMap=map_merge(self.derivedParamMap,          {'b1':self.b1,'b2':self.b2,'b3':self.b3})
    else:
      self.derivedParamMap=map_merge(self.derivedParamMap,          {'b_in':self.b_in,'b_out':self.b_out})
      
    
    self.paramMap=map_merge(self.initParamMap,self.derivedParamMap,{'filter_type':self.filter_type})    
    
    if do_print is True:
      print self.header_str
      print params_to_str(self.paramMap,to_exclude=['summary_str'])
      print
      print self.summary_str
          
    return True
 
      
  def plot_eigs(self):
    import pylab as pl
    from plotlib import custom_axes   
    import matplotlib.ticker as mtick
    pl.figure(figsize=(15,10))   
    pl.subplots_adjust(left=0.05,right=0.95,top=0.9,bottom=0.2,hspace=0.4,wspace=0.4)


    pl.subplot(221)
    custom_axes()
    pl.xlabel('Frequency [1/m]')
    pl.ylabel('Eigenvalue')
    
    pl.plot(self.freqs[1:],self.raw_eigs[1:],'.-k',markersize=10)
    pl.axhline(self.a,color='g',linestyle='-')
    pl.xlim(-0.1,6)
    pl.title('Raw Eigs, LF=%.2f, HF=%.2f '%(self.eigs_lf_diff,self.eigs_hf_diff))  
    

    pl.subplot(222)
    custom_axes()
    pl.xlabel('Frequency [1/m]')
    pl.ylabel('Eigenvalue')
    pl.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    pl.plot(self.freqs[1:],self.eigs[1:]*self.eta,'.-k',markersize=10)
    pl.axhline(0,color='g',linestyle='-')
    pl.xlim(-0.1,6)
    pl.title('Eigs')  
    


    pl.subplot(223)
    t_vect,step_vect,resp_vect,K_vect=get_step_response(self.paramMap)
    pl.plot(t_vect,resp_vect,'-k')
    custom_axes()
    pl.xlabel('Time [s]')
    pl.ylabel('Response [a.u.]')
    pl.xlim(0,3)
    pl.title('Step response, step_amp=10')
    max_resp=resp_vect.max()
    pl.text(2,max_resp*0.8,'tau1=%.2f'%self.tau1)
    pl.text(2,max_resp*0.7,'tau2=%.2f'%self.tau2)
    pl.text(2,max_resp*0.6,'tau3=%.2f'%self.tau3)
    pl.text(2,max_resp*0.4,'mu1=%.2f'%self.mu1)
    pl.text(2,max_resp*0.3,'mu2=%.2f'%self.mu2)
    pl.text(2,max_resp*0.2,'mu3=%.2f'%self.mu3)
   
    pl.subplot(224)
    pl.plot(t_vect,K_vect,'-k')
    custom_axes()
    pl.xlabel('Time [s]')
    pl.ylabel('Response [a.u.]')
    pl.xlim(0,1)
    
  def plot_estimated_initial_output(self):
    
    import pylab as pl      
    from plotlib import noframe,colorbar,shiftedColorMap
    from grid_functions import get_estimated_output
    out_map=get_estimated_output(self.__dict__,self.inputs_flat,self.J0,self.boundary_input_flat)
    pl.figure(figsize=(10,5))
    pl.subplot(121,aspect='equal')
    pl.pcolormesh(out_map.reshape(self.nx,self.nx))
    colorbar()
    noframe()
    pl.title('Initial output')    
    
    pl.subplot(122,aspect='equal')
    data=self.boundary_input_flat.reshape(self.nx,self.nx)
    mesh=pl.pcolormesh(data,vmin=data.min(),vmax=data.max())
    midpoint=1 - data.max()/(data.max() + abs(data.min()))
    cmap = shiftedColorMap(pl.cm.RdBu_r,midpoint=midpoint )
    mesh.set_cmap(cmap)
    colorbar()
    noframe()

    pl.title('Boundary input')    

  def run(self,do_print=True) :
 

    self.snap_idx=0
    self.walk_step_idx=0
    
 
    self.r1=0.
    self.r2=0.
    self.r3=0.
       
    self.r_out=self.r0
    self.rout_av=self.r0
    
    self.J=deepcopy(self.J0)
    self.dJ=np.zeros_like(self.J)
    self.J_vect=np.zeros((self.N,self.num_snaps))
    self.dJ_vect=np.zeros((self.N,self.num_snaps))
    self.tot_input_vect=np.zeros(self.num_snaps)
    self.r_out_vect=np.zeros(self.num_snaps)

    self.final_space_r_out=np.zeros(self.nx**2)
    self.final_space_visits=np.zeros(self.nx**2,dtype=np.int32)
    
    if self.debug_vars is True:    
      
      self.gg_vect=np.zeros((self.N,self.num_snaps))

      self.r1_vect=np.zeros(self.num_snaps)
      self.r2_vect=np.zeros(self.num_snaps)
      self.r3_vect=np.zeros(self.num_snaps)
      
    
    # initial simulation to get rid of the transient
    self.num_step_transient=int(self.transient_time*self.dt)
    self.inner_run(self.num_step_transient,record=True,plastic=False)

    # actual simulation    
    self.inner_run(self.num_sim_steps-self.num_step_transient,record=True,plastic=self.plastic)
    
    
    
    
  def inner_run(self,num_steps,record=False,plastic=True):
            
    progress_clock=clock()
    t=0
    
    # run the simulation
    for step_idx in xrange(num_steps):

      t+=self.dt
      
      # read out input activities at this time step
      if np.remainder(step_idx,self.position_dt_scale)==0:
        
        # if we are at the end of the walk we start again
        if self.walk_step_idx>=self.walk_steps:
          self.walk_step_idx=0

        # read inputs at this walk step        
        self.cur_pos_idx=self.pidx_vect[self.walk_step_idx]
        self.gg=self.inputs_flat[self.cur_pos_idx,:]

        # scale input to keep mean input constant when close to the boundaries
        #self.gg=self.gg/self.gg.mean()*self.input_mean
        
        self.walk_step_idx+=1 

      
      # total input and filtering      
      self.h=np.dot(self.J,self.gg)
      
      self.r1+=self.b1*self.dt*(self.mu1*self.h-self.r1)      
      self.r2+=self.b2*self.dt*(self.mu2*self.h-self.r2)      
      self.r3+=self.b3*self.dt*(self.mu3*self.h-self.r3)      
       
      
      self.r_out=self.r0+self.r1+self.r2+self.r3-self.gamma*self.J.sum()+self.boundary_input_flat[self.cur_pos_idx]
        
      if self.clip_out_rate is True:
        self.r_out=np.clip(self.r_out,0,1000)
              
      
      # in the last 4 snaps  of the simulation record spatial rout
      if self.snap_idx>=self.num_snaps-4:
        self.final_space_r_out[self.cur_pos_idx]+=self.r_out
        self.final_space_visits[self.cur_pos_idx]+=1
        
      # save variables
      if np.remainder(step_idx,self.delta_snap)==0 and record is True:  
        
        print_progress(self.snap_idx,self.num_snaps,progress_clock)
        self.J_vect[:,self.snap_idx]=self.J
        self.dJ_vect[:,self.snap_idx]=self.dJ
        self.r_out_vect[self.snap_idx]=self.r_out
        #self.tot_input_vect[self.snap_idx]=self.gg.mean()

          
        # save debugging variables        
        if self.debug_vars is True:
          self.h_act_vect[self.snap_idx]=self.h_act
          self.h_inact_vect[self.snap_idx]=self.h_inact
          self.gg_vect[:,self.snap_idx]=self.gg
                                 
        self.snap_idx+=1
            
                
      if plastic is True:

        # update weights 
        self.dJ=self.gg*(self.r_out+self.beta-self.alpha*self.J)
        #self.dJ=self.r_out*(self.gg-self.r_out*self.J)            # oja's rule
        
        self.J+=self.dt*self.eta*self.dJ

        # clip weights
        if self.clip_weights is True:
          np.clip(self.J,0.,self.up_bound,out=self.J)
    
  def post_run(self,do_print=True):
      
    # logging simulation end
    endTime=datetime.datetime.fromtimestamp(time.time())
    endTimeStr=endTime.strftime('%Y-%m-%d %H:%M:%S')
    elapsedTime =clock()-self.startClock

    logSim(self.hash_id,'',self.startTimeStr,endTimeStr,elapsedTime,self.paramMap,self.paramsPath,doPrint=False)
      
    if do_print:
      print 'Simulation ends: %s'%endTimeStr
      print 'Elapsed time: %s\n' %format_elapsed_time(elapsedTime)    


    # estimated output rates (boundary input excluded)
    r_vect=np.dot(self.inputs_flat,self.J_vect)
    self.filt_r_vect=filter_r_vect(r_vect,self.paramMap)+self.r0-self.gamma*self.J_vect.sum(axis=0)


    # real final output rate and score   
    #self.final_space_visits[self.final_space_visits==0]=-1
    #self.final_space_r_out_mean=(self.final_space_r_out/self.final_space_visits).reshape(self.nx,self.nx)
  

    # compute final weight and rate scores
    self.final_weights=self.J_vect[:,-1].reshape(self.n,self.n)  
    
    score, spacing,angle,phase,cx= get_grid_params(self.final_weights,
                                                  self.L,self.n,
                                                  num_steps=50,
                                                  return_cx=True)

    self.final_weight_score=score
    self.final_weight_angle=angle
    self.final_weight_spacing=spacing
    self.final_weight_phase=phase
    self.final_weight_cx=cx
        
    self.final_rates=self.filt_r_vect[:,-1].reshape(self.nx,self.nx)  
    
    score, spacing,angle,phase,cx= get_grid_params(self.final_rates,
                                                  self.L,self.nx,
                                                  num_steps=50,
                                                  return_cx=True)

    self.final_rate_score=score
    self.final_rate_angle=angle
    self.final_rate_spacing=spacing
    self.final_rate_phase=phase
    self.final_rate_cx=cx
    
    
    
                
    # save variables
    toSaveMap={
    'paramMap':self.paramMap,
    'J_vect':self.J_vect,
    'dJ_vect':self.dJ_vect,
    'r_out_vect':self.r_out_vect,
    'J0':self.J0,
    'filt_r_vect':self.filt_r_vect,
    #'tot_input_vect':self.tot_input_vect,
    'final_weights':self.final_weights,
    'final_weight_score':self.final_weight_score,
    'final_weight_angle':self.final_weight_angle,
    'final_weight_spacing':self.final_weight_spacing,
    'final_weight_phase':self.final_weight_phase,
    'final_weight_cx':self.final_weight_cx,
    
    'final_rates':self.final_rates,
    'final_rate_score':self.final_rate_score,
    'final_rate_angle':self.final_rate_angle,
    'final_rate_spacing':self.final_rate_spacing,
    'final_rate_phase':self.final_rate_phase,
    'final_rate_cx':self.final_rate_cx
    }
    
    #    'final_space_r_out':self.final_space_r_out,
    #    'final_space_visits':self.final_space_visits,
    #    'final_space_r_out_mean':self.final_space_r_out_mean}
       
    # compute scores   
    if self.compute_scores is True:
      
      if self.inputs_type==InputType.INPUT_GAU_GRID: 
        print 'Computing weight scores'
        # weight scores
        scores,spacings,angles,phases=compute_scores_evo(self.J_vect,self.n,self.L)
        scoresVars={'scores':scores,'spacings':spacings,'angles':angles,'phases':phases}
        toSaveMap=dict(toSaveMap.items() + scoresVars.items()) 
      
      print 'Computing rate scores'
      # rate scores
      rate_scores,rate_spacings,rate_angles,rate_phases=compute_scores_evo(self.filt_r_vect,self.nx,self.L,num_steps=20)    
      scoresVars={'rate_scores':rate_scores,'rate_spacings':rate_spacings,'rate_angles':rate_angles,'rate_phases':rate_phases}
      toSaveMap=dict(toSaveMap.items() + scoresVars.items()) 
      
    # add debug variables for saving
    if self.debug_vars is True:
      #debugVars={'h_act_vect':self.h_act_vect,'h_inact_vect':self.h_inact_vect,'gg_vect':self.gg_vect}
      debugVars={'r1_vect':self.r1_vect,'r2_vect':self.r2_vect,'r3_vect':self.r3_vect,'gg_vect':self.gg_vect}      
      toSaveMap=dict(toSaveMap.items() + debugVars.items())    
          
          
          
    if self.correct_border_effects is True:
      toSaveMap=dict(toSaveMap.items() + {'border_envelope':self.border_envelope}.items())    
            
    # save    
    ensureParentDir(self.dataPath)
    np.savez(self.dataPath,**toSaveMap)
    
    if do_print:
      print 'Result saved in: %s\n'%self.dataPath

    

    
if __name__ == '__main__':
  par_map= map_merge(GridRateParams.gau_grid_small_arena_biphasic_neg,
                    {
                    'r0':10.,
                    'clip_out_rate': True,
                    'periodic_walk':False
                    })
                      
  sim=GridRate(par_map)  
  
  
  sim.post_init()
  if sim.do_run and not run_from_ipython():
    sim.run()
    sim.post_run()
  elif run_from_ipython():
    sim.plot_eigs()
                        

