# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:36:43 2016

@author: dalbis
"""


import numpy as np
from simlib import print_progress
from time import clock
import datetime,time
import os
from grid_inputs import GridInputs
from numpy.fft import fft2,ifft2,fftshift
from numpy.linalg import eigvals
from grid_functions import get_periodic_dist, get_non_periodic_dist,map_merge
from scipy.integrate import romb
from grid_functions import K_t,K_outeq_t,corr_rate
from scipy.integrate import quad
from grid_const import InputType,FilterType
from simlib import ensureParentDir

class GridCorrSpace(object):
  """
  Input correlations in space
  """

  results_path='../results/grid_corr_space'

  @staticmethod
  def get_key_params(paramMap):
  
    filter_key_params_filter_input =['mu1','mu2','mu3','tau1','tau2','tau3']
    filter_key_params_filter_output=['mu_out','tau_in','tau_out']
    
    basic_key_params=GridInputs.get_key_params(paramMap)+['speed',]
    key_params=[]
    if 'filter_type' not in paramMap.keys() or paramMap['filter_type']==FilterType.FILTER_INPUT:
      key_params=basic_key_params+filter_key_params_filter_input
    else:
      key_params=basic_key_params+filter_key_params_filter_output     

    if 'norm_bound_add' in paramMap:
      key_params=key_params+['norm_bound_add',]                 

    if 'norm_bound_mul' in paramMap:
      key_params=key_params+['norm_bound_mul',]                 

    return key_params
    
  @staticmethod
  def get_id(paramMap):
    
    filename=''    
    for param in GridCorrSpace.get_key_params(paramMap):
      filename+='%s=%s_'%(param,paramMap[param])
    filename=filename[:-1]
    return filename
    
  @staticmethod  
  def get_data_path(paramMap):
    return os.path.join(GridCorrSpace.results_path,GridCorrSpace.get_id(paramMap)+'_data.npz')
    
    
  def __init__(self,paramMap,auto_gen=True,do_print=True,force_gen_inputs=False,force_gen_corr=False,keys_to_load=[],use_theory=True,normalize_equal_mean_adding=False,normalize_equal_mean_scaling=False):

    self.filter_type=FilterType.FILTER_INPUT
    self.use_theory=use_theory
  
        
    # import parameters
    for param in GridCorrSpace.get_key_params(paramMap):
      setattr(self,param,paramMap[param])
    
    if 'filter_type' in paramMap.keys():
      self.filter_type=paramMap['filter_type']
                
    self.id=GridCorrSpace.get_id(paramMap)
    self.dataPath=os.path.join(GridCorrSpace.results_path,self.id+'_data.npz')   
    

    if auto_gen is True:   
      
      # generate and save data   
      if force_gen_corr or not os.path.exists(self.dataPath):
        self.gen_data(do_print=do_print,force_gen_inputs=force_gen_inputs)
  
      # load data      
      self.load_data(do_print=do_print,keys_to_load=keys_to_load)
    

  def gen_data(self,do_print=True,force_gen_inputs=False):
    """
    Generates corr space data and saves it to disk
    """

    
    if do_print:
      print
      print 'Generating corr space data, id = %s'%self.id
    
    self.post_init(force_gen_inputs=force_gen_inputs)
    self.run()
    self.post_run()
    
    
  def load_data(self,do_print=True,keys_to_load=[]):
    """
    Loads data from disk
    """
    
    if do_print:
      print
      print 'Loading corr_space data, Id = %s'%self.id


    data= np.load(self.dataPath,mmap_mode='r')
    
    loaded_keys=[]
    
    if len(keys_to_load)==0:
      for k,v in data.items():
        setattr(self,k,v)
        loaded_keys.append(k)
    else: 
      for k in keys_to_load:
        setattr(self,k,data[k])
        loaded_keys.append(k)

     
    if do_print:
      print 'Loaded variables: '+' '.join(loaded_keys)
      
    
  def post_init(self,force_gen_inputs=False):
    
    self.startClock=clock()
    self.startTime=datetime.datetime.fromtimestamp(time.time())
    self.startTimeStr=self.startTime.strftime('%Y-%m-%d %H:%M:%S')
    

    self.dx=self.L/self.nx
    X,Y=np.mgrid[-self.L/2:self.L/2:self.dx,-self.L/2:self.L/2:self.dx]
    self.pos=np.array([np.ravel(X), np.ravel(Y)]).T

    if self.filter_type==FilterType.FILTER_INPUT:    
      self.b1=1./self.tau1
      self.b2=1./self.tau2
      self.b3=1./self.tau3
    elif self.filter_type==FilterType.FILTER_OUTPUT:
      self.b_in=1./self.tau_in
      self.b_out=1./self.tau_out
      

    
    self.N=self.n**2
    
    # number of samples for the filter
    self.tau_samps=2**8+1
    self.tau_ran=np.arange(self.tau_samps)*self.dx
    
    if self.filter_type==FilterType.FILTER_INPUT:
      self.K_samp= K_t(self.b1,self.b2,self.b3,self.mu1,self.mu2,self.mu3,self.tau_ran/self.speed)/self.speed
    elif self.filter_type==FilterType.FILTER_OUTPUT:
      self.K_samp= K_outeq_t(self.b_in,self.b_out,self.mu_out,self.tau_ran/self.speed)/self.speed
    
    tapered=False
    if hasattr(self,'tap_inputs') and self.tap_inputs is True:
      tapered=True
      
    
    # for Gaussian periodic we just compute analytically     
    if self.inputs_type == InputType.INPUT_GAU_GRID \
       and self.use_theory is True and tapered is False:

      self.compute_analytically=True
      print 'Analytical estimation for Gaussian receptive fields (%s)'%('periodic' if self.periodic_inputs else 'non periodic')
      
      ran,step=np.linspace(-self.L/2.,self.L/2.,self.n,endpoint=False,retstep=True)      
      SSX,SSY = np.meshgrid(ran,ran)
      self.centers= np.array([np.ravel(SSX), np.ravel(SSY)]).T
      self.amp=self.input_mean*self.L**2/(2*np.pi*self.sigma**2)
      
    else:

      print 'Numerical estimation for general inputs'
      self.compute_analytically=False

      # load inputs 
      self.inputs=GridInputs(self.__dict__,force_gen=force_gen_inputs)
      self.inputs_flat=self.inputs.inputs_flat
      self.inputs_path=self.inputs.dataPath

    
    # parameters map
    self.paramMap = {'id':self.id,
                'L':self.L,'n':self.n,'speed':self.speed,'nx':self.nx,
                'sigma':self.sigma,'input_mean':self.input_mean,
                'periodic_inputs':self.periodic_inputs,
                'inputs_type':self.inputs_type}
                
    if self.filter_type == FilterType.FILTER_INPUT:
      self.paramMap=map_merge(self.paramMap, {'tau1':self.tau1,'tau2':self.tau2,'tau3':self.tau3,
                'mu1':self.mu1,'mu2':self.mu2,'mu3':self.mu3,
                'b1':self.b1,'b2':self.b2,'b3':self.b3})
    elif self.filter_type == FilterType.FILTER_OUTPUT:
       self.paramMap=map_merge(self.paramMap, {'tau_in':self.tau_in,'tau_out':self.tau_out,'mu_out':self.mu_out,
                                               'b_in':self.b_in,'b_out':self.b_out})
  

  #@profile  
  def run(self):


    # initialize CC matrix
    CC_teo=np.zeros((self.N,self.N))


    # analytical computation for periodic gaussians
    if self.compute_analytically is True:
      
      # choose the K filter function depending on the filter type (input or equivalent output)
      if self.filter_type==FilterType.FILTER_INPUT:
        K_t_fun=lambda t: K_t(self.b1,self.b2,self.b3,self.mu1,self.mu2,self.mu3,t)
      elif self.filter_type==FilterType.FILTER_OUTPUT:
        K_t_fun=lambda t: K_outeq_t(self.b_in,self.b_out,self.mu_out,t)
        

      # shorthand for the analytical correlation function
      corr_rate_short=lambda tau,u:  corr_rate(K_t_fun,self.speed,self.sigma,tau,u)

      # compyute distance matrix
      CC_dist=np.zeros_like(CC_teo)
      
      if self.periodic_inputs is True:
        get_dist=get_periodic_dist
      else:
        get_dist=get_non_periodic_dist
      
      for i in xrange(self.N):
        for j in xrange(self.N):
          CC_dist[i,j]= get_dist(self.centers[i,:],self.centers[j,:],self.L)


      # fill in correlation value for each distance
      all_dist=np.unique(CC_dist.ravel())      
      for dist in all_dist:      
        corr=np.pi*self.amp**2*self.sigma**2/self.L**2*quad(corr_rate_short, 0.,2.,args=(dist))[0]
        CC_teo[CC_dist==dist]=corr
    
    
    # numerical calculation for general inputs                          
    else:
      
      #pyfftw.interfaces.cache.enable()
      
      # compute DFTs  
      #inputs_mat=pyfftw.empty_aligned((self.nx,self.nx,self.N), dtype='float32')
      #inputs_mat[:]=self.inputs_flat.reshape(self.nx,self.nx,self.N)
      inputs_mat=self.inputs_flat.reshape(self.nx,self.nx,self.N)
      inputs_dfts=fft2(inputs_mat,axes=[0,1])
          
      
      # binning for line integral
      center=np.array([self.nx,self.nx])/2
      yr, xr = np.indices(([self.nx,self.nx]))
      r = np.around(np.sqrt((xr - center[0])**2 + (yr - center[1])**2)).astype(int)
      nr = np.bincount(r.ravel())
    
      snap_idx=0 
      prog_clock=clock()
      num_snaps=self.N*(self.N+1)/2
    
      # loop over matric elements    
      for i in xrange(self.N):
       
        input_i_dft=inputs_dfts[:,:,i]
      
        for j in xrange(i,self.N):
          
          print_progress(snap_idx,num_snaps,start_clock=prog_clock,step=num_snaps/100)
  
            
          # get j-input
          input_j_dft=inputs_dfts[:,:,j]
              
          # inputs correlation      
          dft_prod=input_i_dft*np.conj(input_j_dft)
          
          #dft_prod=pyfftw.empty_aligned((self.nx,self.nx), dtype='complex64')
          #dft_prod[:]=input_i_dft*np.conj(input_j_dft)

          input_corr=fftshift(np.real(ifft2(dft_prod)))*self.dx**2
        
          # integral on a circle of radius tau  
          count = np.bincount(r.ravel(), input_corr.ravel())/nr  
          corr_prof_teo=np.zeros(len(self.tau_ran))  
          corr_prof_teo[:self.nx/2] = count[:self.nx/2]    
          
          # convolution with the filter
          CC_teo[i,j]=romb(self.K_samp*corr_prof_teo,dx=self.dx)

          snap_idx+=1

      # normalize and fill upper triangle
      CC_teo/=self.L**2    
      CC_teo=CC_teo+CC_teo.T
      CC_teo[np.diag(np.ones(self.N).astype(bool))]*=0.5    
          
    
     
    # normalize correlation matrix to equal mean at the boundary (additive approach)
    if self.periodic_inputs is False and hasattr(self,'norm_bound_add') and self.norm_bound_add is True:
      print 'Correlation matrix boundary normalization (additive)'
      C4d=CC_teo.reshape(self.n,self.n,self.n,self.n)
      C4d_norm=np.zeros_like(C4d)
      mean_C=C4d.mean(axis=3).mean(axis=2)  
      for i in xrange(self.n):
        for j in xrange(self.n):
          C4d_norm[i,j,:,:]=C4d[i,j,:,:]-mean_C[i,j]+mean_C.min()
      C_norm=C4d_norm.reshape(self.N,self.N) 
 
      CC_teo=C_norm

       
    # normalize correlation matrix to equal mean at the boundary (multiplicative approach)
    if self.periodic_inputs is False and hasattr(self,'norm_bound_mul') and self.norm_bound_mul is True:
      print 'Correlation matrix boundary normalization (multiplicative)'
      C4d=CC_teo.reshape(self.n,self.n,self.n,self.n)
      C4d_norm=np.zeros_like(C4d)
      mean_C=C4d.mean(axis=3).mean(axis=2)  
      for i in xrange(self.n):
        for j in xrange(self.n):
          C4d_norm[i,j,:,:]=C4d[i,j,:,:]/mean_C[i,j]*mean_C.min()
      C_norm=C4d_norm.reshape(self.N,self.N) 
 
      CC_teo=C_norm


          
    self.CC_teo=CC_teo    
    self.eigs=eigvals(self.CC_teo)

  def post_run(self):
    
    ensureParentDir(self.dataPath)
    np.savez(self.dataPath,paramMap=self.paramMap,
          CC_teo=self.CC_teo,
          eigs=self.eigs,computed_analytically=self.compute_analytically)
          
