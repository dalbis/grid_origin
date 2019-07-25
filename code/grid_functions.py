

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:37:16 2016

@author: dalbis
"""

import numpy as np
from scipy.integrate import quad,dblquad
from scipy.special import iv,jv
import glob
import os
from collections import namedtuple
from scipy.signal import fftconvolve
from grid_const import InputType,FilterType

def map_merge(*args):
  tot_list=[]
  for arg in args:
    tot_list=tot_list+arg.items()    
  return dict(tot_list)

# ==== difference of exponentials
  
K_t_sign = lambda   b1,b2,b3,mu1,mu2,mu3,t: 0.5*(np.sign(t)+1)*K_t(b1,b2,b3,mu1,mu2,mu3,t)

K_t= lambda b1,b2,b3,mu1,mu2,mu3,t:\
 (mu1*b1*np.exp(-t*b1)+mu2*b2*np.exp(-t*b2)+mu3*b3*np.exp(-t*b3))

K_ft_k= lambda b1,b2,b3,mu1,mu2,mu3,k:\
 mu1*b1/(b1+1j*k)+mu2*b2/(b2+1j*k)+mu3*b3/(b3+1j*k)

# hankel transform of the kernel in space (divided by tau and weighted by speed)  
K_ht_k = lambda b1,b2,b3,mu1,mu2,mu3,speed,k: \
  mu1*(b1/speed)/(k**2+(b1/speed)**2)**0.5 \
+ mu2*(b2/speed)/(k**2+(b2/speed)**2)**0.5 \
+ mu3*(b3/speed)/(k**2+(b3/speed)**2)**0.5


# equivalent output kernel in time
def K_outeq_t(b_in,b_out,mu_out,t):
  g=b_out*(1-mu_out)  
  c1=b_in/(b_in-g)*(b_in-b_out)
  c2=b_in/(b_in-g)*mu_out*b_out
  
  return c1*np.exp(-t*b_in)+c2*np# normalization factor for mixture of gaussians



def K_outeq_t_sign(b_in,b_out,mu_out,t): 
  return 0.5*(np.sign(t)+1)*K_outeq_t(b_in,b_out,mu_out,t)
  
  
# fourier transform of the equivalent output kernel 
def K_outeq_ft_k(b_in,b_out,mu_out,k):
  g=b_out*(1-mu_out)  
  c1=b_in/(b_in-g)*(b_in-b_out)
  c2=b_in/(b_in-g)*mu_out*b_out

  return c1/(b_in+1j*k)+c2/(g+1j*k)

# hankel transform of the equivalent output kernel in space 
def K_outeq_ht_k(b_in,b_out,mu_out,speed,k):

  g=b_out*(1-mu_out)  
  c1=b_in/(b_in-g)*(b_in-b_out)
  c2=b_in/(b_in-g)*mu_out*b_out
  
  return   (c1/speed)/(k**2+(b_in/speed)**2)**0.5 \
         + (c2/speed)/(k**2+(g/speed)**2)**0.5

# ===== difference of gaussians 
Kg_t = lambda s1,s2,mu,t:\
  np.exp(-t**2/(2*s1**2))-mu*np.exp(-t**2/(2*s2**2))

kg_h_k = lambda s1,s2,mu,k:\
   2*np.pi*(np.exp(-k**2*s1**2/2.)*s1**2-mu*np.exp(-k**2*s2**2/2.)*s2**2)
  
  
# ========== gaussian input
g_x = lambda amp,sigma,x: amp*np.exp(-x**2/(2*sigma**2)) 
g_ht_k = lambda amp,sigma,k: 2*np.pi*amp*sigma**2*np.exp(-k**2*sigma**2/2.)



teo_scale_factor_pw= lambda n: np.pi/(3*n)*(4/np.pi+1./(3*n))
    
# theoretical correlation function for gaussian inputs
corr_rate = lambda K_t_fun,speed,sigma,tau,u: K_t_fun(tau)*np.exp(-(u**2+(tau*speed)**2)/(4*sigma**2))*iv(0,tau*speed*u/(2*sigma**2))
    

def get_estimated_output(paramMap,inputs_flat,J,boundary_input_flat):
    from numpy import dot,real

    from numpy.fft import fftshift,fft2,ifft2

    p=get_params(paramMap)
    r=dot(inputs_flat,J)
    
    K_ft=get_spatial_filter(paramMap)    
    r_mat_dft=fftshift(fft2(r.reshape(p.nx,p.nx)),)  
    filt_r_mat_ft=r_mat_dft*K_ft
    filt_r_mat=real(ifft2(fftshift(filt_r_mat_ft)))    
    filt_r=filt_r_mat.reshape(p.nx**2)
  
    out_map=filt_r+p.r0-p.gamma*p.J_av_star*p.N+boundary_input_flat
    return  out_map
    
def get_params(paramMap):
  Params = namedtuple('Params', paramMap.keys())
  return Params(**paramMap)  



def get_step_response(paramMap,step_start=0.5,step_stop=6,step_amp=10.,t_min=0,t_max=6.,f_max=5.):  

  p=get_params(paramMap)

  dt=0.001   
  t_vect=np.arange(t_min,t_max,dt)  


  if p.filter_type==FilterType.FILTER_INPUT:
    b1=1/p.tau1
    b2=1/p.tau2
    b3=1/p.tau3
    K_vect=K_t_sign(b1,b2,b3,p.mu1,p.mu2,p.mu3,t_vect)  
  else:
    b_in=1./p.tau_in
    b_out=1./p.tau_out
    K_vect=K_outeq_t_sign(b_in,b_out,p.mu_out,t_vect)  
  
  step_fun = lambda t,t0: (np.sign(t-t0)+1.)*0.5   
  step_vect=step_amp*(step_fun(t_vect,step_start)-step_fun(t_vect,step_stop))
  resp_vect=fftconvolve(step_vect,K_vect,mode='full')[:len(t_vect)]*dt
  
  return t_vect,step_vect,resp_vect,K_vect


def get_non_periodic_dist(c1,c2,L):
  dux=(c1-c2)[0]
  duy=(c1-c2)[1]

  return np.sqrt(dux**2+duy**2)

# periodic distance between two gaussian centers
def get_periodic_dist(c1,c2,L):
  c1=c1+L/2.
  c2=c2+L/2.

  dux=(c1-c2)[0]
  duy=(c1-c2)[1]
  
  ux = min(abs(dux),L-abs(dux))
  uy = min(abs(duy),L-abs(duy))

  rho_u=np.sqrt(ux**2+uy**2)
  return rho_u
  
  
def compute_Cmean(CC_mean,ret_var=True):
  
  """
  Compute average correlation matrix and its DFT
  """
  N=CC_mean.shape[0]
  n=int(np.sqrt(N))
  
  # mean correlation and its DFT         
  C4d=CC_mean.reshape(n,n,n,n)    
  C4d_centered=np.zeros_like(C4d)
  for i in xrange(n):
    for j in xrange(n):
       C4d_centered[i,j,:,:]=np.roll(np.roll(C4d[i,j,:,:],-i+n/2,axis=0),-j+n/2,axis=1)

  C_mean=np.mean(C4d_centered.reshape(N,n,n),axis=0)
  C_var=np.var(C4d_centered.reshape(N,n,n),axis=0)
  C_mean_real_ft=np.real(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(C_mean)))).astype(float)
  
  C4d_homo=np.zeros_like(C4d)
  for i in xrange(n):
    for j in xrange(n):
       C4d_homo[i,j,:,:]=np.roll(np.roll(C_mean,i-n/2,axis=0),j-n/2,axis=1)
       
  C_homo=C4d_homo.reshape(N,N)
  
  if ret_var:      
    return C_mean,C_var,C_mean_real_ft,C_homo
  else:
    return C_mean
    




def compute_teo_eigs(inputs,paramMap,teo_input_pw=False,freqs=np.arange(0,6,0.001)):
  """
  Compute theoretical eigenvalues for the given inputs and the model parameters in paramMap
  For non-Gaussian inputs the average input power is computed numerically, therefore the
  eigenvalues and their frequencies are returned in discrete steps of size 1/L
  """

  p=get_params(paramMap)
  amp=p.input_mean*p.L**2/(2*np.pi*p.sigma**2)

  if not hasattr(p,'filter_type') or p.filter_type==FilterType.FILTER_INPUT:
    b1=1./p.tau1
    b2=1./p.tau2
    b3=1./p.tau3
    
  else:
    b_in=1./p.tau_in
    b_out=1./p.tau_out
    
  # gaussian grid  
  if teo_input_pw is True:
    assert(p.inputs_type in [InputType.INPUT_GAU_GRID, InputType.INPUT_GAU_MIX_POS])
    in_freqs=freqs
    
    if p.inputs_type == InputType.INPUT_GAU_GRID or \
       (p.inputs_type == InputType.INPUT_GAU_MIX_POS and p.num_gau_mix==1):
       
      input_pw=g_ht_k(amp,p.sigma,2*np.pi*in_freqs)**2
      
    else:
      input_pw=g_ht_k(amp,p.sigma,2*np.pi*in_freqs)**2*teo_scale_factor_pw(p.num_gau_mix)

  # all other cases
  else:    
    in_freqs=inputs.in_freqs
    input_pw=inputs.in_mean_pw_profile    

  # input filter
  if not hasattr(p,'filter_type') or p.filter_type==FilterType.FILTER_INPUT:
    K_ht=K_ht_k(b1,b2,b3,p.mu1,p.mu2,p.mu3,p.speed,2*np.pi*in_freqs)
  
  # output filter  
  elif p.filter_type==FilterType.FILTER_OUTPUT:
    K_ht=K_outeq_ht_k(b_in,b_out,p.mu_out,p.speed,2*np.pi*in_freqs)
            
 
  corr_ht_est=input_pw*K_ht/p.L**2
  
  density=p.n**2/p.L**2
  eigs=corr_ht_est*density
  
  return in_freqs,eigs

def get_spatial_filter(paramMap,plot=False):
  
  p=get_params(paramMap)

  
  freqs=np.fft.fftshift(np.fft.fftfreq(int(p.nx),p.L/p.nx))
  fX,fY=np.meshgrid(freqs,freqs)
  f_vect=2*np.pi*np.array([np.ravel(fX), np.ravel(fY)]).T
  
  k_vect_norm=np.sqrt(f_vect[:,0]**2+f_vect[:,1]**2)

  if not hasattr(p,'filter_type') or p.filter_type==FilterType.FILTER_INPUT:      
    K_ht_flat=K_ht_k(p.b1,p.b2,p.b3,p.mu1,p.mu2,p.mu3,p.speed,k_vect_norm)
  else:
    K_ht_flat=K_outeq_ht_k(p.b_in,p.b_out,p.mu_out,p.speed,k_vect_norm)
  
  
  K_ht=K_ht_flat.reshape(p.nx,p.nx)

  if plot:
    import pylab as pl
    pl.figure()
    pl.subplot(111,aspect='equal')
    pl.pcolormesh(fX,fY,K_ht)
    pl.colorbar()
  return K_ht

  
def filter_r_vect(r_vect,paramMap):
  """
  Filters the total weighted input over time with a spatial filter that is equivalent to the temporal filter. 
  The baseline rate r0 is also added. This is an estimate of the real output rate of the neuron over time. 
  """
  
  p=get_params(paramMap)

  
  K_ft=get_spatial_filter(paramMap)
  
  r_mat=r_vect.reshape(p.nx,p.nx,p.num_snaps)
  r_mat_dft=np.fft.fftshift(np.fft.fft2(r_mat,axes=[0,1]),axes=[0,1])

  
  filt_r_mat_ft=r_mat_dft*K_ft[:,:,np.newaxis]
  filt_r_mat=np.real(np.fft.ifft2(np.fft.fftshift(filt_r_mat_ft,axes=[0,1]),axes=[0,1]))
  
  filt_r_vect=filt_r_mat.reshape(p.nx**2,p.num_snaps)
  return filt_r_vect


  
    
def compute_gaussian_teo_corr(rate,paramMap,quad_int=True,compute_matrices=True,d_w=0.01,fmax=6):
  """
  Computes input correlation function, its hankel transform. 
  
  Example usage:
  
  C_mean,C_mean_ft,corr_prof,corr_prof_ht,uran,fran,kran=compute_gaussian_teo_corr(rate_model,paramMap,fmax=6)
      
  # compute eigenvalues 
  if rate_model is True:
    eigs=(C_mean_ft-a)*eta
    eigs_prof=(corr_prof_ht*density-a)*eta
  else:
    eigs=(C_mean_ft+k3)*eta
    eigs_prof=(corr_prof_ht+k3)*eta
  
  """

  #print 'Computing theoretical correlations'
  p=get_params(paramMap)

  amp=p.input_mean*p.L**2/(2*np.pi*p.sigma**2)
    
  # do not allow naive integration with compute_matrices
  assert(quad_int is True or (quad_int is False and compute_matrices is False))
  
  # kernel
    
  if not hasattr(p,'filter_type') or p.filter_type==FilterType.FILTER_INPUT:
    K_t_fun = lambda t: K_t(p.b1,p.b2,p.b3,p.mu1,p.mu2,p.mu3,t)
  else:
    K_t_fun = lambda t: K_outeq_t(p.b_in,p.b_out,p.mu_out,t)
    
  corr_rate_fun =lambda tau,u:  corr_rate(K_t_fun,p.speed,p.sigma,tau,u)
    
  
  # integration limits and steps
  d_u=0.01
  d_s=0.005
  d_tau=0.001
  
  u_min,u_max=0,2
  tau_min,tau_max=0,2
  w_min,w_max=0,2*np.pi*fmax
  s_min,s_max=-.5,.5


  if compute_matrices is True:  
    
    # compute the distances of all place field centers
    ran,step=np.linspace(-p.L/2.,p.L/2.,p.n,endpoint=False,retstep=True)
    UX,UY = np.meshgrid(ran,ran)
    allu = np.sqrt(UX**2 +UY**2)

    # compute theoretical correlation matrix
    C_mean=np.zeros((p.n,p.n))
    samp_uran=np.unique(allu.ravel())

  
  # RATE BASED
  if rate is True:

      
    # quad integration
    if quad_int is True:

      # profile      
      corr_prof=np.pi*(amp**2)*p.sigma**2/(p.L**2)*np.array([quad(corr_rate_fun,tau_min,tau_max,args=(u)) for u in np.arange(u_min,u_max,d_u)])[:,0]
      
      # matrix
      if compute_matrices is True:
        for u in samp_uran:
          c=np.pi*(amp**2)*p.sigma**2/(p.L**2)*quad(corr_rate_fun,tau_min,tau_max,args=(u))[0]
          C_mean[allu==u]=c
    
    # naif integration    
    else:
      U2,T2=np.mgrid[u_min:u_max:d_u,tau_min:tau_max:d_tau]
      A2=K_t_fun(T2)*np.exp(-(U2**2+T2**2)/(4*p.sigma**2))*iv(0,T2*U2/(2*p.sigma**2))
      corr_prof=np.pi*(amp**2)*p.sigma**2/(p.L**2)*sum(A2,1)*d_tau
  
  
  # SPIKE BASED
  else:

    # theoretical function
    step_pos   = lambda t: (np.sign(t)+1)*0.5
    step_neg   = lambda t: (1-np.sign(t))*0.5
    W_fun_t    = lambda t: step_neg(t)*p.Aplus*np.exp(t/p.tau_plus)+step_pos(t)*p.Aminus*np.exp(-t/p.tau_minus)
    corr_spike_fun = lambda s,tau,u: W_fun_t(s)*K_t_fun(tau)*np.exp(-(u**2+(tau-s)**2)/(4*p.sigma**2))*iv(0,(tau-s)*u/(2*p.sigma**2))  
  
    # quad integration
    if quad_int is True:
      
      # profile
      corr_prof=np.pi*(amp**2)*p.sigma**2/(p.L**2)*np.array([dblquad(corr_spike_fun,tau_min,tau_max,lambda s: s_min,lambda s: s_max,args=(u,)) for u in np.arange(u_min,u_max,d_u)])[:,0]
      
      # matrix 
      if compute_matrices is True:
        for u in samp_uran:
          c=np.pi*(amp**2)*p.sigma**2/(p.L**2)*dblquad(corr_spike_fun,tau_min,tau_max,lambda s: s_min,lambda s: s_max,args=(u,))[0]
          C_mean[allu==u]=c
      
    # naive integration
    else:
      U3,T3,S3=np.mgrid[u_min:u_max:d_u,tau_min:tau_max:d_tau,s_min:s_max:d_s]
      A3=W_fun_t(S3)*K_t(T3)*np.exp(-(U3**2+(T3-S3)**2)/(4*p.sigma**2))*iv(0,(T3-S3)*U3/(2*p.sigma**2))
      corr_prof=np.pi*(amp**2)*p.sigma**2/(p.L**2)*sum(sum(A3,2),1)*d_tau*d_s
    
    
  # FOURIER DOMAIN
    
  # hankel transform of the profile curve
  W2,U2=np.mgrid[w_min:w_max:d_w,u_min:u_max:d_u]
  corr_prof_ht=2*np.pi*sum(corr_prof[np.newaxis,:]*U2*jv(0,W2*U2),1)*d_u#*(p.n**2)/p.L**2

  # compute matrix dft        
  if compute_matrices is True:
    C_mean_ft=np.real(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(C_mean)))).astype(float)

  uran=np.arange(u_min,u_max,d_u)  
  fran=np.arange(w_min,w_max,d_w)/(2*np.pi)
  kran=np.fft.fftshift(np.fft.fftfreq(int(p.n),d=p.L/float(p.n)))

  if compute_matrices is False:
    return corr_prof,corr_prof_ht,uran,fran,kran
  else:
    return C_mean,C_mean_ft,corr_prof,corr_prof_ht,uran,fran,kran
    

def select_data(data_folder,pick_first=False):
  """
  Select data to load from a list of available files
  """

  files=glob.glob(data_folder+'*.npz')
  files.sort(key=lambda x: -os.path.getmtime(x))
  
  print 'Data in %s'%(data_folder)
  print '============================================'
  for idx,path in enumerate(files):
    fname=path.replace(data_folder,'')
    print '%d) %s'%(idx,fname)
  print
  
  if pick_first is False:  
    sel = raw_input('Index [0]: ')
    sel_idx = 0 if sel=='' else int(sel)
  else:
    sel_idx=0
  
  return files[sel_idx]


def load_data(data_path):
  """
  Load data to a target dictionary
  """
  
  # load data
  print 'Loading: %s'%data_path
  assert(os.path.exists(data_path))

  data=np.load(data_path,mmap_mode='r')
  
  # load parameters into locals
  paramMap=data['paramMap'][()]
  p=get_params(paramMap)

  Results = namedtuple('Params', data.keys())
  r=Results(**data)
  
  

  print '==========    DATA    =========='
  print '\n'.join(data.keys())
  print
  
  return p,r
  
