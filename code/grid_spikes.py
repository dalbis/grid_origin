# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:40:29 2015

@author: dalbis
"""

from numpy import exp,array,arange,sum,\
ones,zeros,clip,dot,mean,\
remainder,savez,ascontiguousarray,hstack,ones_like,around,unique,squeeze,newaxis,zeros_like
from numpy.random import randn,seed
from simlib import params_to_str,logSim,gen_string_id,gen_hash_id
from simlib import run_from_ipython,format_elapsed_time, ensureParentDir
from time import clock
import datetime,time
import os
import brian2 as b
from grid_inputs import GridInputs
from grid_walk import GridWalk
from gridlib import compute_scores_evo
from grid_functions import K_t,map_merge,filter_r_vect
from grid_params import GridSpikeParams
from grid_const import InputType
from gridlib import get_grid_params



class GridSpikes:
  """
  2D spiking model
  """
  
  key_params=['inputs_type','n','sim_time','eta','a','J_av_target','J0_std','seed',
              'gamma','tau1','tau2','tau3','mu1','mu2','mu3','r0']

  
  results_path = '../results/grid_spikes'
    
  def __init__(self,paramMap):
    
        
    # general parameters
    self.n=None
    self.dt=None
    self.num_snaps=None
    self.eta=None
    self.sim_time=None
    self.seed=None

    # walk parameters   
    self.L=None
    self.speed=None
    self.theta_sigma = None
    self.walk_time=None
    self.walk_seed=None        
    self.periodic_walk=None
    self.bounce=None
    self.nx=None
    
    # input parameters
    self.periodic_inputs=None
    self.sigma=None
    self.input_mean=None
    self.inputs_type=None
    self.centers_std=None
    self.inputs_seed=None
    self.outside_ratio=None
    
    # adaptation kernel in time
    self.tau1=None
    self.tau2=None
    self.tau3=None
    
    self.mu1=None
    self.mu2=None
    self.mu3=None
    
    # eigenvalue threshold and desired average weight    
    self.a=None
    self.J_av_target=None
    self.gamma=None
    
    self.up_bound=None 
    self.J0_std=None 
    self.r0=None
    
    # STDP
    self.Aplus = None
    self.Aminus = None 
    self.tau_plus = None
    self.tau_minus = None
        
    # parameters for average weight update
    self.average_weight_update =False
    self.average_weight_update_dt=2.5
        
    # set parameter values from input map
    for param,value in paramMap.items():
      setattr(self,param,value)

    # parameters we never change
    self.plastic=True
    self.test_neuron_mode=False
    
    
    self.transient_time=2.
    self.arena_shape='square'
    self.virtual_bound_ratio=1.0
    self.bounce_theta_sigma=0.0
    self.position_dt=self.L/self.nx    
    self.profile=False

    self.constant_inputs=False
    self.plastic=True    
    self.use_stdp=True
    self.clip_weights=True               
    self.record=True
    self.method='euler'
    
    
    # sanity checks
    if self.periodic_walk is True:
      assert(self.bounce==False)
      assert(self.virtual_bound_ratio==1.0)
      assert(self.outside_ratio==1.0)
      
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
                'tau1':self.tau1,'tau2':self.tau2,'tau3':self.tau3,
                'mu1':self.mu1,'mu2':self.mu2,'mu3':self.mu3,
                'eta':self.eta,'a':self.a,'gamma':self.gamma,
                'Aplus':self.Aplus,'Aminus':self.Aminus,
                'tau_plus':self.tau_plus,'tau_minus':self.tau_minus,
                'J0_std':self.J0_std,'up_bound':self.up_bound,'r0':self.r0,
                'J_av_target':self.J_av_target,
                'clip_weights':self.clip_weights,
                'periodic_inputs':self.periodic_inputs,
                'inputs_type':self.inputs_type,
                'inputs_seed':self.inputs_seed,
                'walk_seed':self.walk_seed,'walk_time':self.walk_time,
                'periodic_walk':self.periodic_walk,
                'bounce':self.bounce,'bounce_theta_sigma':self.bounce_theta_sigma,
                'compute_scores':self.compute_scores,
                'outside_ratio':self.outside_ratio,
                'virtual_bound_ratio':self.virtual_bound_ratio,
                'use_stdp':self.use_stdp,                
                'plastic':self.plastic,
                'variable_speed':self.variable_speed,
                'speed_theta':self.speed_theta,
                'speed_sigma':self.speed_sigma
                }
                
                
    # human-readable parameter strings (just for printing)            
    self.key_params_str=params_to_str(self.initParamMap,keyParams=GridSpikes.key_params,compact=True)            
    self.input_params_str=params_to_str(self.initParamMap,keyParams=GridInputs.get_key_params(self.initParamMap),compact=True)
    self.walk_params_str=params_to_str(self.initParamMap,keyParams=GridWalk.key_params,compact=True)
    
    # generate id and paths
    self.str_id=gen_string_id(self.initParamMap)
    self.hash_id=gen_hash_id(self.str_id)
   
    self.paramsPath=os.path.join(GridSpikes.results_path,self.hash_id+'_log.txt')
    self.dataPath=os.path.join(GridSpikes.results_path,self.hash_id+'_data.npz')
    self.figurePath=os.path.join(GridSpikes.results_path,self.hash_id+'_fig.png')
    
    
    ##### FORCING
    if os.path.exists(self.dataPath) and False:
      #print 'Data hash %s already present'%self.hash_id
      self.do_run=False
    else:
      self.do_run=True  
          
      
  def post_init(self,do_print=True):
    
          
    seed(self.seed)  
        
    # monitors
    self.monitors=[]

    self.startClock=clock()
    self.startTime=datetime.datetime.fromtimestamp(time.time())
    self.startTimeStr=self.startTime.strftime('%Y-%m-%d %H:%M:%S')
    
    # inverse time constants
    #self.b0=1/self.tau_epsp
    self.b1=1/self.tau1
    self.b2=1/self.tau2
    self.b3=1/self.tau3
    
    self.N=self.n**2


    # integral of the STDP learning window W
    self.W0 = self.Aplus*self.tau_plus+self.Aminus*self.tau_minus    
    
    # epsp and adaptation kernel
    W_plus_t   = lambda t: self.Aplus*exp(-t/self.tau_plus)
    W_minus_t  = lambda t: self.Aminus*exp(-t/self.tau_minus)


    # intgral of the squared learning window
    ddt=0.00001
    tran=arange(0,.3,ddt)
    self.W_squared_0 = (sum(W_plus_t(tran)**2)+sum(W_minus_t(tran)**2))*ddt

    # compute the filter H, i.e., the convolution between adaptation kernel and epsp
    ddt=0.00001
    tran=arange(0,4,ddt)
    K_vect=K_t(self.b1,self.b2,self.b3,self.mu1,self.mu2,self.mu3,tran)
    self.K_vect=K_vect
    
    # integral of the filter    
    self.K0=self.mu1+self.mu2+self.mu3
    

    # compute the integral of the positive side of W by H
    self.W_K=dot(self.Aplus*exp(-tran/self.tau_plus),K_vect)*ddt
    
    self.C_av=self.input_mean**2*self.K0*self.W0
    
    self.B=(self.a-self.N*self.C_av)*self.J_av_target
    self.beta=self.B/self.input_mean-self.r0*self.W0
    self.alpha=self.a/self.input_mean+self.W_K

    # No STDP window
    if self.use_stdp is False:
      self.W0=0.
      self.C_av=0.
      self.W_K=0.
      self.W_squared_0=0.
      
      
    self.k1=self.input_mean*(self.beta+self.r0*self.W0)
    self.k2=self.input_mean*self.W0*self.gamma
    self.k3=self.input_mean*(self.alpha-self.W_K)    
    
    self.tau_av=1./(self.eta*(self.a-self.N*(self.C_av-self.input_mean*self.gamma)))

   
    self.J_av_star=self.B*self.eta*self.tau_av

    self.J0_mean=self.J_av_star

    # average output rate
    self.r_out_star=(self.input_mean*self.K0-self.gamma)*self.J_av_star*self.N+self.r0

    # diffusion time constant  
    self.D1=self.input_mean*self.beta**2
    self.D2=self.input_mean*self.r_out_star*self.W_squared_0
    self.D3=self.input_mean*self.r_out_star*self.W0*(2*self.beta+self.W0*(self.input_mean+self.r_out_star))               
    self.D=self.eta**2*(self.D1+self.D2+self.D3)

    self.D4=self.input_mean*self.r_out_star*(2*self.beta*self.W0+self.r_out_star*self.W0**2)
    self.Dprime=self.eta**2*(self.D1+self.D2+self.D3+self.D4)
    
    # noise time constant
    self.tau_noise=self.J_av_star**2/self.D
        
    self.num_sim_steps = int(self.sim_time/self.dt)
    self.delta_snap = int(self.num_sim_steps/self.num_snaps)

    assert(self.delta_snap*self.num_snaps==self.num_sim_steps)
    

    # initial weight    
    self.J0=ones(self.N)*self.J0_mean+randn(self.N)*self.J0_std
    self.J0=self.J0/mean(self.J0)*self.J0_mean
    self.J0=clip(self.J0,0,self.up_bound)


    # load walk data
    walk=GridWalk(self.initParamMap)
    self.walk_path=walk.dataPath
    self.pos=walk.pos
    self.pidx_vect=walk.pidx_vect
    self.nx=walk.nx
    self.walk_steps=walk.walk_steps

    # get input rates
    if self.constant_inputs is False:
      
      inputs=GridInputs(self.initParamMap)   #__dict__ 
      self.inputs_path=inputs.dataPath
      self.inputs_flat=inputs.inputs_flat
    else:
      
      self.inputs_path='NONE'
      

                
    self.derived_param_str = 'alpha=%.2f beta=%.2f B=%.2f tau_av=%.1e tau_noise=%.1e r_out_star=%.2f'%(self.alpha,self.beta,self.B,self.tau_av,self.tau_noise,self.r_out_star)

    self.summary_str=  """
    
HASH: %s

KEY PARAMS: %s

INPUT PARAMS: %s

WALK PARAMS: %s

DERIVED PARAMS: %s
      
      """%(self.hash_id,self.key_params_str,self.input_params_str,self.walk_params_str,self.derived_param_str)            

 
    # parameters map
    self.derivedParamMap = {
                'hash_id':self.hash_id,'delta_snap':self.delta_snap,
                'beta':self.beta,'alpha':self.alpha,
                'B':self.B,
                'tau_av':self.tau_av,
                'k1':self.k1,'k2':self.k2,'k3':self.k3,
                'D1':self.D1,'D2':self.D2,'D3':self.D3,'D4':self.D4,'D':self.D,'Dprime':self.Dprime,
                'r_out_star':self.r_out_star,'tau_noise':self.tau_noise,'C_av':self.C_av,
                'W0':self.W0,'K0':self.K0,'W_squared_0':self.W_squared_0, 'W_K':self.W_K,
                'J0_mean':self.J0_mean,'J0_std':self.J0_std,'J_av_star':self.J_av_star,
                'b1':self.b1,'b2':self.b2,'b3':self.b3,
                'summary_str':self.summary_str
                }
                
    
    self.paramMap=dict(self.initParamMap.items()+self.derivedParamMap.items())

                

    if do_print is True:  
      print
      print '================================================================='
      print '                 DETAILED SPIKING SIMULATION                     '
      print '================================================================='
      print
      
      print params_to_str(self.paramMap,to_exclude=['summary_str'])
      print
      print self.summary_str    
     
     
  def run(self,do_print=True):
        
    global r0,up_bound,Aplus_eta,Aminus_eta,tau_plus,tau_minus,beta_eta,alpha_eta,\
    b0,b1,b2,b3,mu1,mu2,mu3,tau1,tau2,tau3,gamma
    
    b.prefs.codegen.target = 'cython'
    b.prefs.codegen.cpp.extra_compile_args_gcc = ['-O3']

    # additional variable for testing neuron functionality
    if self.test_neuron_mode is True:
      self.r1=0
      self.r2=0
      self.r3=0

  
      self.gg_vect=zeros((self.N,self.num_sim_steps))
      self.r_out_vect=zeros(self.num_sim_steps)
            
      self.J=zeros(self.N)
      
    self.J=self.J0
    
    # local variables for Brian simulation    
    r0=self.r0*b.Hz
    b1=self.b1*b.Hz
    b2=self.b2*b.Hz
    b3=self.b3*b.Hz

    mu1=self.mu1
    mu2=self.mu2
    mu3=self.mu3
                
    tau_plus=self.tau_plus*b.second
    tau_minus=self.tau_minus*b.second
    tau1=self.tau1*b.second
    tau2=self.tau2*b.second
    tau3=self.tau3*b.second
    
    gamma=self.gamma*b.Hz
    
    up_bound=self.up_bound

    # absorbe learning rate eta
    Aplus_eta=self.Aplus*self.eta
    Aminus_eta=self.Aminus*self.eta

    beta_eta=self.beta*self.eta
    alpha_eta=self.alpha*self.eta
    
    self.walk_step_idx=0
    self.step_idx=0    
    self.snap_idx=0
    
    if self.test_neuron_mode is True or self.constant_inputs is False:
      self.gg=self.inputs_flat[self.pidx_vect[self.walk_step_idx],:]
          
    self.J_vect=zeros((self.N,self.num_snaps))
    self.num_low_weights_vect=zeros(self.num_snaps)
    
    self.num_low_weights=0
    self.num_low_weights_count=0
  
    @b.network_operation(dt=self.delta_snap/100.*self.dt*b.second)
    def count_low_weights():
      self.num_low_weights+=sum(self.synapses.w[:,:]<3*self.eta)
      self.num_low_weights_count+=1
              
              
    @b.network_operation(dt=self.delta_snap*self.dt*b.second)
    def save_weights():
      self.J_vect[:,self.snap_idx]=self.synapses.w[:,:]
      self.num_low_weights_vect[self.snap_idx]=\
      0 if self.num_low_weights_count==0 else  self.num_low_weights/self.num_low_weights_count
      self.num_low_weights=0
      self.num_low_weights_count=0
      self.snap_idx+=1
      
            
    # function to update rat position  
    @b.network_operation(dt=self.position_dt*b.second)
    def update_position():
      """
      Updates the position of the virtual rat at every simulation time step
      """
      
      # if we are at the end of the walk we start again
      if self.walk_step_idx>=self.walk_steps:
        self.walk_step_idx=0

      # read inputs at this walk step        
      self.gg=self.inputs_flat[self.pidx_vect[self.walk_step_idx],:]
      self.walk_step_idx+=1 

      self.input_group.rates=self.gg*b.Hz


    # function to update rat position  
    @b.network_operation(dt=self.average_weight_update_dt*b.second)
    def average_weight_update():
      """
      Updates the weights by taking the average changes over seconds
      """
      self.synapses.w[:,:]+=self.synapses.w_temp[:,:]
      self.synapses.w[:,:].clip(min=0)
      self.synapses.w_temp[:,:]*=0
        
    @b.network_operation(dt=self.dt*b.second)
    def test_neuron_update():    
      self.h=dot(self.J,self.gg)      
      self.r1+=self.b1*self.dt*(self.mu1*self.h-self.r1)      
      self.r2+=self.b2*self.dt*(self.mu2*self.h-self.r2)      
      self.r3+=self.b3*self.dt*(self.mu3*self.h-self.r3)      
      self.r_out=self.r0+self.r1+self.r2+self.r3-self.gamma*self.J.sum()
      
    

      if self.record is True:        
        self.gg_vect[:,self.step_idx]=squeeze(self.gg)
        self.r_out_vect[self.step_idx]=self.r_out
        self.step_idx+=1
          
    # Poisson input neurons  
    if self.constant_inputs is True:
      inputGroup=b.PoissonGroup(self.N,rates=self.input_mean*b.Hz,dt=self.dt*b.second)
    else:                
      inputGroup=b.PoissonGroup(self.N,rates=self.gg*b.Hz,dt=self.dt*b.second)

      
    self.input_group=inputGroup    
    

    eqs= """
      v=r0+r1+r2+r3-gamma*wtot :Hz
      dr1/dt=-b1*r1 :Hz
      dr2/dt=-b2*r2 :Hz
      dr3/dt=-b3*r3 :Hz
      wtot :1
      """
      
    outputGroup=b.NeuronGroup(1,model=eqs,threshold='rand()<v*dt',method=self.method,name='output_neuron',dt=self.dt*b.second)
    
    # add monitors if we are testing neuron functioning 
    if self.test_neuron_mode is True:
      self.output_monitor = b.SpikeMonitor(outputGroup,name='output_monitor')
      self.input_monitor = b.SpikeMonitor(inputGroup,name='input_monitor')
      self.v_monitor = b.StateMonitor(outputGroup,'v', when='end',record=True,name='v_monitor')

      self.monitors.append(self.output_monitor)
      self.monitors.append(self.input_monitor)
      self.monitors.append(self.v_monitor)


    synapse_eqs=[]
    synapse_pre_eqs=[]
    synapse_post_eqs=[]
    
    # equations to drive output neuron
    synapse_pre_eqs.append('r1+=mu1*b1*w')
    synapse_pre_eqs.append('r2+=mu2*b2*w')
    synapse_pre_eqs.append('r3+=mu3*b3*w')
         
    # synapse equation
    synapse_eqs.append('w: 1')
    
    
    if self.average_weight_update is True:
      synapse_eqs.append('w_temp: 1')
      
    synapse_eqs.append('wtot_post = w : 1  (summed)')                    
                    
    # plastic synapses 
    if self.plastic is True:      
      
      # non Hebbian term
      
      if self.average_weight_update is True:
        synapse_pre_eqs.append('w_temp+=(beta_eta-alpha_eta*w)')
      else:
        synapse_pre_eqs.append('w+=(beta_eta-alpha_eta*w)')
        
      # STDP Hebbian terms
      if self.use_stdp is True:
        
        # update traces
        synapse_eqs.append('dApre/dt=-Apre/tau_plus : 1 (event-driven)')
        synapse_eqs.append('dApost/dt=-Apost/tau_minus : 1 (event-driven)')

        # every pre-synaptic spike      
        synapse_pre_eqs.append('Apre+=Aplus_eta')        
        if self.average_weight_update is True:
          synapse_pre_eqs.append('w_temp+=Apost')
        else:
          synapse_pre_eqs.append('w+=Apost')

        # every post-synaptic spike
        synapse_post_eqs.append('Apost+=Aminus_eta')
        if self.average_weight_update is True:
          synapse_post_eqs.append('w_temp+=Apre')
        else:        
          synapse_post_eqs.append('w+=Apre')
      
    # clip weights  
    if self.clip_weights is True and self.average_weight_update is False:
      synapse_pre_eqs.append('w=clip(w,0,up_bound)')
      synapse_post_eqs.append('w=clip(w,0,up_bound)')
 
    
    if do_print is True:
      print
      print 'Synapse model: ',synapse_eqs
      print 'Synapse pre: ', synapse_pre_eqs
      print 'Synapse post: ', synapse_post_eqs
      print    
        
    # create synapses
    synapses=b.Synapses(inputGroup,outputGroup,
                        model='\n'.join(synapse_eqs),
                        pre='\n'.join(synapse_pre_eqs),
                        post='\n'.join(synapse_post_eqs),name='synapses',method=self.method,dt=self.dt*b.second)
    synapses.connect(True)
    synapses.w[:,:]=self.J0
    if self.average_weight_update is True:
      synapses.w_temp[:,:]=zeros_like(self.J0)
    
    self.synapses=synapses
    

    # create the basic network 
    self.network=b.Network()
    self.network.add(inputGroup,outputGroup,synapses)

    # count low weigts
    #self.network.add(count_low_weights)
    
    # add position update network operation
    if self.constant_inputs is False:
      self.network.add(update_position)    
      
    if self.average_weight_update is True:
      self.network.add(average_weight_update)

    # add network function for testing
    if self.test_neuron_mode is True:
      self.network.add(test_neuron_update)
          
    # add monitors to the network      
    for mon in self.monitors:
      self.network.add(mon)

    self.network.add(save_weights)
        
    # first 2 seconds to get rid of the transient     
    self.network.run(self.transient_time*b.second)

    # real simulation    
    synapses.w[:,:]=self.J0
    self.network.run((self.sim_time-self.transient_time)*b.second,report='text',profile=self.profile)

    if self.profile is True and do_print is True:
      print
      print b.profiling_summary(net=self.network)
      print

    

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
    r_vect=dot(self.inputs_flat,self.J_vect)
    self.filt_r_vect=filter_r_vect(r_vect,self.paramMap)+self.r0-self.gamma*self.J_vect.sum(axis=0)

  

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
    'J0':self.J0,
    'filt_r_vect':self.filt_r_vect,
    'num_low_weights_vect':self.num_low_weights_vect,
    
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
         
          
    # save     
    ensureParentDir(self.dataPath)
    savez(self.dataPath,**toSaveMap)
    
    if do_print:
      print 'Result saved in: %s\n'%self.dataPath
      
    

      
  
def plot_neuron_test(sim):
  """
  Some basic plotting to test the neuron functionality
  """
  
  input_spikes=sim.input_monitor.num_spikes
  output_spikes=sim.output_monitor.num_spikes

  teo_input_spikes=sum(sum(sim.gg_vect,1)*sim.dt)
  teo_output_spikes=sum(squeeze(sim.v_monitor.v[:]))*sim.dt
  
  print 'input_spikes: %d/%d'%(input_spikes,teo_input_spikes)
  print 'output_spikes: %d/%d'%(output_spikes,teo_output_spikes)
  
  offset=0.
  time=arange(int(sim.sim_time/sim.dt))*sim.dt
  
  import pylab as pl
  from plotlib import custom_axes,colorbar
  
  # input rates and spikes
  pl.figure()
  pl.pcolormesh(time,arange(sim.N),sim.gg_vect,cmap='Greys')
  colorbar()
  for i in xrange(sim.N):
    st=b.asarray(sim.input_monitor.spike_trains()[i])-offset
    pl.plot(st,ones_like(st)*i,'.k')
  custom_axes()
  pl.xlabel('Time [s]')
  pl.ylabel('Input num.')
  

  
  allst=[]
  for i in xrange(sim.N):
    allst.extend(b.asarray(sim.input_monitor.spike_trains()[i]).tolist())
  pl.title('Input_spikes   Expected=%d    Count=%d'%(teo_input_spikes,len(allst)))

  
  # output rates and spikes
  pl.figure(figsize=(12,5))
  
  pl.subplot(111)
  pl.plot(time,squeeze(sim.r_out_vect),'-k')
  st=b.asarray(sim.output_monitor.spike_trains()[0])-offset
  pl.plot(st,ones_like(st)*10,'.k')
  pl.plot(time,squeeze(sim.v_monitor.v),'-b')
  pl.xlabel('Time [s]')
  pl.ylabel('Output rate')
  pl.title('Output_spikes   Expected=%d    Count=%d'%(teo_output_spikes,len(st)))
  custom_axes()
  

def plot_output_spikes(sim):
  import pylab as pl
  import plotlib as pp
  import numpy as np
  # output rates and spikes
  
  time=arange(int(sim.sim_time/sim.dt))*sim.dt


  pl.figure(figsize=(12,5))
  
  pl.subplot(111)
  #pl.plot(time,squeeze(sim.r_out_vect),'-k')
  s_times=np.sort(b.asarray(sim.output_monitor.spike_trains()[0]))
  s_int=np.diff(s_times)
  
  voltage=squeeze(sim.v_monitor.v)
  
  for s_time in s_times:
    st_idx=np.where(time>s_time)[0][0]
    pl.plot([s_time, s_time],[voltage[st_idx],20],'-k')
  
  pl.plot(time,voltage,'-k')
  pl.xlim(0,10)
  pl.xlabel('Time [s]')
  pl.ylabel('Output rate')
  #pl.title('Output_spikes   Expected=%d    Count=%d'%(teo_output_spikes,len(st)))
  pp.custom_axes()
  
  pl.figure()
  pl.hist(s_int)
  
def plot_isi_hist(sim):
  
  import pylab as pl
  import plotlib as pp
  import numpy as np
  
  time=arange(int(sim.sim_time/sim.dt))*sim.dt
  freqs=np.fft.fftfreq(len(time),sim.dt)
  
  voltage=squeeze(sim.v_monitor.v)

  voltage_pw=abs(np.fft.fft(voltage)*sim.dt)**2
  voltage_pw[freqs==0]=0

  

  s_times=np.round(np.sort(b.asarray(sim.output_monitor.spike_trains()[0]))*1000).astype(int)
  bin_train=np.zeros_like(time)
  bin_train[s_times]=10 
  bin_pw=abs(np.fft.fft(bin_train)*sim.dt)**2
  bin_pw[freqs==0]=0


  
  pl.figure()
  pl.subplot(121)
  pl.plot(freqs,voltage_pw)
  pl.xlim(-5,5)
  pp.custom_axes()
  
  pl.subplot(122)
  pl.plot(freqs,bin_pw)
  pl.xlim(-5,5)
  pp.custom_axes()
  
if __name__ == '__main__'  :

  sim = GridSpikes(map_merge(GridSpikeParams.gau_grid_small_arena_biphasic_neg,
                                 {'a':1.1}))                                
  if sim.do_run and (sim.test_neuron_mode or not run_from_ipython()) :
    sim.post_init()
    sim.run()
    
    if not sim.test_neuron_mode:
      sim.post_run()
 
  elif run_from_ipython() and  sim.test_neuron_mode:
    plot_neuron_test(sim)     
    plot_output_spikes(sim)
    plot_isi_hist(sim) 
  
  
  
