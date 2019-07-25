# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 09:33:42 2016

@author: dalbis
"""



from numpy.fft import fft2,fftshift,ifft2,fftfreq
from numpy.random import randn,rand,seed,permutation,randint
import numpy as np
import os
from simlib import print_progress, gen_hash_id,gen_string_id
from time import clock
from gridlib import compute_scores_evo,dft2d_profiles,detect_peaks
from grid_const import InputType
from simlib import ensureParentDir

def gen_perfect_grids(pos,n,grid_T,grid_angle,
                      angle_sigma=0,grid_T_sigma=0,
                      jitter_axes_angle=False,
                      jitter_axes_T=False):
                        
  """
  Generates n**2 regular grids evenly distributed in phase space with scale
  grid_T and orientation grid_angle

  If angle_sigma>0 the orientations are jittered accordingly
  If T_sigma > 0 the grid scale is jittered accordingly
  If jitter_axes is True the three grid axes are jittered independently

  """                      


  if jitter_axes_T is True:
     T_cos_vects=[]                                                                 # 3, Nx1
     for i in xrange(3):
       grid_T_vect=np.ones((n**2,1))*grid_T+grid_T_sigma*np.random.randn(n**2,1)    # Nx1 
       T_cos_vect = grid_T_vect/2*np.sqrt(3)                                        # Nx1
       T_cos_vects.append(T_cos_vect)
    
  else:
     grid_T_vect=np.ones((n**2,1))*grid_T+grid_T_sigma*np.random.randn(n**2,1)    # Nx1 
     T_cos_vect = grid_T_vect/2*np.sqrt(3)                                        # Nx1
     T_cos_vects = [ T_cos_vect for i in xrange(3)]                               # 3, Nx1


  # unit vectors of the direct lattice
  u1 = grid_T*np.array([np.sin(2*np.pi/3+grid_angle), -np.cos(2*np.pi/3+grid_angle)])
  u2 = grid_T*np.array([-np.sin(grid_angle), np.cos(grid_angle)])
    
  # phase samples
  ran = np.array([np.arange(-n/2.,n/2.)/n]).T
  u1_phases = np.array([u1])*ran
  u2_phases = np.array([u2])*ran
    
  X1,X2=np.meshgrid(u1_phases[:,0],u2_phases[:,0])
  Y1,Y2=np.meshgrid(u1_phases[:,1],u2_phases[:,1])
  X,Y=X1+X2,Y1+Y2

  phases = np.array([np.ravel(X), np.ravel(Y)]).T
  
  
  if jitter_axes_angle is True:
    noise_angles=[angle_sigma*np.random.randn(n**2,1) for i in xrange(3)]  # 3,Nx1 
  else:
    na=angle_sigma*np.random.randn(n**2,1)
    noise_angles=[na for i in xrange(3)]  # 3,Nx1 
  
  angles=[np.pi*i/3+grid_angle+noise_angles[i] for i in xrange(3)]            # 3, N
  ks = [ 2*np.pi/T_cos_vects[i]*np.hstack((np.cos(angles[i]),np.sin(angles[i]))) for i in xrange(3)] # 3, Nx2
  
  
  angle_vect=np.array(angles)
  
 
  pos_x = pos[:,0]
  pos_y = pos[:,1]
  
  phases_x = phases[:,0]
  phases_y = phases[:,1]
  
  pp_x = pos_x[:,np.newaxis]+phases_x[np.newaxis,:]
  pp_y = pos_y[:,np.newaxis]+phases_y[np.newaxis,:]
  
  grid=np.zeros((pos.shape[0],n**2))
  
 
  for k in ks:
    grid=grid+(np.cos(k[:,0]*pp_x+k[:,1]*pp_y))
    
  return phases,grid,angle_vect,grid_T_vect   
  
  
  
def add_noise_and_normalize(pos,nx,N,noise_sigma,input_mean,signal_weight,grids):
  
  if signal_weight < 1.:
    
    # generate Gaussian noise
    sigma=1
    xi_raw=(sigma*randn(nx,nx,N))  # mean: sigma*sqrt(2)/sqrt(pi)  var: sigma**2*(1-2/pi) 
        
    # Gaussian filter with unit integral
    filt_x=np.exp(-np.sum(pos**2,1)/(2*noise_sigma**2))
    filt_x/=filt_x.sum()
      
    # Filter noise
    filt_x=filt_x.reshape(nx,nx)
    filt_x_ft=fft2(filt_x)
    xi_ft = fft2(xi_raw)
    xi_filt_ft=np.multiply(xi_ft,filt_x_ft[:,:,np.newaxis])
    xi = np.real(ifft2(xi_filt_ft,axes=[0,1]))
    xi=xi.reshape(nx**2,N)
  
    # clip and fix the mean  
    grids=np.clip(grids,0,100.)
    B=input_mean/grids.mean(axis=0)
    grids*=B
  
         
    # normalize signal and noise with same mean and variance
    xi=xi*np.sqrt(grids.var(axis=0))/np.sqrt(xi.var(axis=0))
    xi=xi-np.mean(xi,axis=0)+input_mean
    
    # weighted mean of signal and noise  
    inputs=signal_weight*grids+(1-signal_weight)*xi

  else:
    inputs=grids
    xi=np.zeros_like(grids)

    
  # clip and normalize the mean back 
  inputs=np.clip(inputs,0,100.)
  inputs=inputs/inputs.mean(axis=0)*input_mean
  inputs_flat=np.ascontiguousarray(inputs,dtype=np.float32)
  

  xi=np.clip(xi,0,100.)
  xi=xi/xi.mean(axis=0)*input_mean  
  noise_flat=np.ascontiguousarray(xi,dtype=np.float32)
  
  grids_flat=np.ascontiguousarray(grids,dtype=np.float32)
      
  return inputs_flat,grids_flat,noise_flat
    

# detect peaks coordinated from a regular grids and jitter the peaks 
def get_noisy_peaks(grid,X,Y,scatter_sigma):
    
  peaks_raw=detect_peaks(grid,size=2)  
  peaks=np.bitwise_and(peaks_raw,grid>0.99*grid.max())


  # filtering out duplicate peaks (neighboring pixels)    
  peaks_xy_dup = np.array([X[peaks==1],Y[peaks==1]])
  
  from scipy.spatial.distance import pdist,squareform
  dists=pdist(peaks_xy_dup.T, 'euclidean')
  alldists=squareform(dists)

  num_raw_peaks=len(alldists)
  peaks_idxs=range(num_raw_peaks)
      
  for i in xrange(num_raw_peaks):
    for j in range(i+1,num_raw_peaks):
      if j in peaks_idxs and alldists[i,j]<0.05:
        peaks_idxs.remove(j)

  num_peaks=len(peaks_idxs)
  peaks_xy=np.zeros((2,num_peaks))
  
  for i in xrange(num_peaks):
    peaks_xy[:,i]=peaks_xy_dup[:,peaks_idxs[i]]
      
  peaks_xy_noisy=peaks_xy+np.random.randn(2,peaks_xy.shape[1])*scatter_sigma
  return peaks_xy_noisy

    
  
class GridInputs(object):
  
  results_path='../results/grid_inputs'
  basic_key_params=['inputs_type','n','nx','L','periodic_inputs','input_mean','sigma']
  
  key_params_map={
  
    InputType.INPUT_GAU_GRID: basic_key_params+['outside_ratio'],
    InputType.INPUT_GAU_NOISY_CENTERS: basic_key_params+['centers_std','inputs_seed','outside_ratio'],
    InputType.INPUT_GAU_RANDOM_CENTERS: basic_key_params+['inputs_seed','outside_ratio'],

    InputType.INPUT_GAU_MIX: basic_key_params+['num_gau_mix','inputs_seed'],  
    InputType.INPUT_GAU_MIX_NEG: basic_key_params+['num_gau_mix','inputs_seed'],  
    InputType.INPUT_GAU_MIX_POS: basic_key_params+['num_gau_mix','inputs_seed'],  
    InputType.INPUT_GAU_MIX_POS_FIXAMP: basic_key_params+['num_gau_mix','inputs_seed'],  
    InputType.INPUT_RAND: basic_key_params+['inputs_seed'],        
    InputType.INPUT_RAND_CORR: basic_key_params+['inputs_seed'],

    InputType.INPUT_BVC: basic_key_params+['sigma_rad_0','beta','sigma_ang'],

    InputType.INPUT_NOISY_GRID: ['inputs_type','n','nx','L','signal_weight',
                                 'grid_angle','grid_T','input_mean','noise_sigma','inputs_seed'],

    InputType.INPUT_NOISY_GRID_TWO_ANGLES: ['inputs_type','nx','L','signal_weight',
                                            'n1','n2','grid_angle1','grid_angle2',
                                            'grid_T','input_mean','noise_sigma','inputs_seed'],
                                            
                                            
    InputType.INPUT_NOISY_GRID_JITTER: ['inputs_type','n','nx','L','signal_weight',
                                 'grid_angle','grid_T','input_mean','noise_sigma','inputs_seed',
                                 'angle_sigma','jitter_axes_angle',
                                 'grid_T_sigma','jitter_axes_T'],
                                                                  
    InputType.INPUT_NOISY_GRID_SCATTERED_FIELDS: ['inputs_type','n','nx','L',
                                                  'grid_angle','grid_T','input_mean','scatter_sigma','inputs_seed',
                                                  'angle_sigma','jitter_axes_angle',
                                                  'grid_T_sigma','jitter_axes_T'],
      

  }

  @staticmethod
  def get_key_params(paramMap):
    return GridInputs.key_params_map[paramMap['inputs_type']]
    
  @staticmethod
  def get_id(paramMap):
    return gen_hash_id(gen_string_id(paramMap,key_params=GridInputs.get_key_params(paramMap)))
    
  @staticmethod  
  def get_data_path(paramMap):
    return os.path.join(GridInputs.results_path,GridInputs.get_id(paramMap)+'_data.npz')
    
    
  def __init__(self,paramMap,do_print=True,comp_scores=True,force_gen=False,keys_to_load=[]):

    keyParams=GridInputs.get_key_params(paramMap)
    
    # import parameters
    for param in keyParams:
      setattr(self,param,paramMap[param])
    
    self.id=GridInputs.get_id(paramMap)
    self.paramsPath=os.path.join(self.results_path,self.id+'_log.txt')
    self.dataPath=os.path.join(self.results_path,self.id+'_data.npz')   

    
    if force_gen or not os.path.exists(self.dataPath):
      
      # generate and save data
      self.gen_data(do_print,comp_scores=comp_scores)

    # load data 
    self.load_data(do_print,keys_to_load)
        

        
  def load_data(self,do_print,keys_to_load=[]):
    """
    Loads data from disk
    """
    
    if do_print:
      print
      print 'Loading input data, Id = %s'%self.id
    
    try:
      data= np.load(self.dataPath,mmap_mode='r')
    except:
      print 'Cannot load %s'%self.dataPath
        
    
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
    


  def gen_data(self,do_print,comp_scores=True):
    """
    Generates input data and saves it to disk
    """
  
    if do_print:  
      print
      print 'Generating inputs data, id = %s'%self.id
      
      print 'Computing scores: %s'%str(comp_scores)
    
    # set seed
    if self.inputs_type not in (InputType.INPUT_GAU_GRID, InputType.INPUT_BVC)  :
      seed(self.inputs_seed)
    
    # calculate additional paramters
    
    if self.inputs_type==InputType.INPUT_NOISY_GRID_TWO_ANGLES: 
      self.N=self.n1**2+self.n2**2
    else:
      self.N=self.n**2
    
    self.dx=self.L/self.nx
    self.X,self.Y=np.mgrid[-self.L/2:self.L/2:self.dx,-self.L/2:self.L/2:self.dx]
    self.pos=np.array([np.ravel(self.X), np.ravel(self.Y)]).T
    
    toSaveMap={'inputs_type':self.inputs_type}

    if hasattr(self,'sigma'):
      # gaussian peak
      self.amp=self.input_mean*self.L**2/(2*np.pi*self.sigma**2)
      
    # noisy grids (grids plus noise)  
    if self.inputs_type==InputType.INPUT_NOISY_GRID:
      self.gen_inputs_noisy_grids(comp_scores)
      toSaveMap.update({'inputs_flat':self.inputs_flat,'input_mean':self.input_mean,'phases':self.phases})
      if comp_scores is True:
        scoresMap={'in_scores':self.in_scores,'in_spacings':self.in_spacings,'in_angles':self.in_angles,
                   'in_phases':self.in_phases}    
        toSaveMap.update(scoresMap)

     # noisy grids (grids plus noise, two different angles)  
    if self.inputs_type==InputType.INPUT_NOISY_GRID_TWO_ANGLES:
      self.gen_inputs_noisy_grids_two_angles(comp_scores)
      toSaveMap.update({'inputs_flat':self.inputs_flat,'input_mean':self.input_mean,'phases':self.phases})
      if comp_scores is True:
        scoresMap={'in_scores':self.in_scores,'in_spacings':self.in_spacings,'in_angles':self.in_angles,
                   'in_phases':self.in_phases}    
        toSaveMap.update(scoresMap)
        

    # noisy grids (grids plus noise, and jittered scales and/or orientations)  
    if self.inputs_type==InputType.INPUT_NOISY_GRID_JITTER:
      self.gen_inputs_noisy_grids_jitter(comp_scores)
      toSaveMap.update({'inputs_flat':self.inputs_flat,
                        'grids_flat':self.grids_flat,
                        'noise_flat':self.noise_flat,
                        'input_mean':self.input_mean,'phases':self.phases,
                        'grid_T_vect':self.grid_T_vect,'angle_vect':self.angle_vect})
      
      if comp_scores is True:
        scoresMap={'in_scores':self.in_scores,'in_spacings':self.in_spacings,'in_angles':self.in_angles,
                   'in_phases':self.in_phases}    
        toSaveMap.update(scoresMap)


    # noisy grids (scattered fields, and jittered scales and/or orientations)   
    if self.inputs_type==InputType.INPUT_NOISY_GRID_SCATTERED_FIELDS:
      self.gen_inputs_noisy_grids_scattered_fields(comp_scores)
      toSaveMap.update({'inputs_flat':self.inputs_flat,'input_mean':self.input_mean,'phases':self.phases,
                        'grid_T_vect':self.grid_T_vect,'angle_vect':self.angle_vect})
      if comp_scores is True:
        scoresMap={'in_scores':self.in_scores,'in_spacings':self.in_spacings,'in_angles':self.in_angles,
                   'in_phases':self.in_phases}    
        toSaveMap.update(scoresMap)
 


    # gaussian inputs
    elif self.inputs_type in (InputType.INPUT_GAU_GRID, InputType.INPUT_GAU_NOISY_CENTERS, InputType.INPUT_GAU_RANDOM_CENTERS):
      self.gen_inputs_gaussian()
      toSaveMap.update({'inputs_flat':self.inputs_flat,'amp':self.amp,'centers':self.centers})

    # mixture of gaussians
    elif self.inputs_type in (InputType.INPUT_GAU_MIX,InputType.INPUT_GAU_MIX_NEG, InputType.INPUT_GAU_MIX_POS,InputType.INPUT_GAU_MIX_POS_FIXAMP,InputType.INPUT_RAND,InputType.INPUT_RAND_CORR):
      self.gen_inputs_random()
      toSaveMap.update({'inputs_flat':self.inputs_flat,'input_scalings':self.input_scalings,'random_amps':self.random_amps})
            
    # boundary vector cells        
    elif self.inputs_type == InputType.INPUT_BVC:
      self.gen_inputs_bvc()
      toSaveMap.update({'inputs_flat':self.inputs_flat})

    # compute spectra
    self.comp_input_spectra()
    spectraMap={'in_freqs':self.in_freqs,'in_mean_dft':self.in_mean_dft,'in_mean_amp':self.in_mean_amp,
    'in_pw_profiles':self.in_pw_profiles,'in_mean_pw_profile':self.in_mean_pw_profile,
    'in_amp_profiles':self.in_amp_profiles,'in_mean_amp_profile':self.in_mean_amp_profile}     
    toSaveMap.update(spectraMap)

    # compute scores    
    if comp_scores is True:
      self.comp_scores()
      scoresMap={'in_scores':self.in_scores,'in_spacings':self.in_spacings,
      'in_angles':self.in_angles,'in_phases':self.in_phases}    
      toSaveMap.update(scoresMap)
 
 
    print toSaveMap.keys()
    
    # save    
    ensureParentDir(self.dataPath)
    np.savez(self.dataPath,**toSaveMap)      
    
    
    
  def comp_input_spectra(self):
    """
    Compute input power spectra
    """
    
    self.nx=int(self.nx)

    
    inputs_mat=self.inputs_flat.reshape(self.nx,self.nx,self.N)

    in_allfreqs = fftshift(fftfreq(self.nx,d=self.L/self.nx))
    self.in_freqs=in_allfreqs[self.nx/2:]
    
    in_dft_flat=fftshift(fft2(inputs_mat,axes=[0,1]),axes=[0,1])*(self.L/self.nx)**2

    in_pw=abs(in_dft_flat)**2
    in_amp=abs(in_dft_flat)
      
    self.in_mean_dft=np.mean(in_dft_flat,axis=2)
    self.in_mean_amp=np.mean(in_amp,axis=2)
    
    self.in_amp_profiles=dft2d_profiles(in_amp)
    self.in_pw_profiles=dft2d_profiles(in_pw)
    self.in_mean_amp_profile=np.mean(self.in_amp_profiles,axis=0)
    self.in_mean_pw_profile=np.mean(self.in_pw_profiles,axis=0)
    
    
  def comp_scores(self):

    self.in_scores,self.in_spacings,self.in_angles,self.in_phases=compute_scores_evo(
    self.inputs_flat,self.nx,self.L,num_steps=50)
  
    
  def gen_inputs_noisy_grids(self,comp_scores=True):
    """
    Generates regular grids and adds correlated Gaussian noise on top
    """
    # generate perfect grids  
    seed(self.inputs_seed)
    self.phases,grids,angle_vect,grid_T_vect=gen_perfect_grids(self.pos,self.n,self.grid_T,self.grid_angle)
    self.inputs_flat,grids_flat,noise_flat=add_noise_and_normalize(self.pos,self.nx,self.N,self.noise_sigma,
                                                  self.input_mean,self.signal_weight,grids)

    # compute scores
    if comp_scores:
      self.in_scores,self.in_spacings,self.in_angles,self.in_phases=compute_scores_evo(
      self.inputs_flat,self.nx,self.L,num_steps=50)
           

    
  def gen_inputs_noisy_grids_jitter(self,comp_scores=True):
    """
    Generates regular grids with nosy orientations and adds correlated Gaussian noise on top
    Grid scale and orientations are jittered (Gaussian distributed)
    """
    # generate perfect grids  
    seed(self.inputs_seed)
    self.phases,grids,self.angle_vect,self.grid_T_vect=\
                                        gen_perfect_grids(self.pos,self.n,self.grid_T,self.grid_angle,
                                        self.angle_sigma,jitter_axes_angle=self.jitter_axes_angle,
                                        grid_T_sigma=self.grid_T_sigma,jitter_axes_T=self.jitter_axes_T)
                                        
    self.inputs_flat,self.grids_flat,self.noise_flat=add_noise_and_normalize(self.pos,self.nx,self.N,
                                                                             self.noise_sigma,
                                                  self.input_mean,self.signal_weight,grids)

    # compute scores
    if comp_scores:
      self.in_scores,self.in_spacings,self.in_angles,self.in_phases=compute_scores_evo(
      self.inputs_flat,self.nx,self.L,num_steps=50)
           
           
  def gen_inputs_noisy_grids_two_angles(self,comp_scores=True):
    """
    Generates regular grids and adds correlated Gaussian noise on top
    """
    # generate perfect grids  
    seed(self.inputs_seed)
    phases1,grids1,angle_vect1,grid_T_vect1=gen_perfect_grids(self.pos,self.n1,self.grid_T,self.grid_angle1)
    phases2,grids2,angle_vect2,grid_T_vect2=gen_perfect_grids(self.pos,self.n2,self.grid_T,self.grid_angle2)
    
    self.phases=np.vstack([phases1,phases2])
    grids=np.hstack([grids1,grids2])
    
    self.inputs_flat,grids_flat,noise_flat=add_noise_and_normalize(self.pos,self.nx,self.N,self.noise_sigma,
                                                  self.input_mean,self.signal_weight,grids)

    # compute scores
    if comp_scores:
      self.in_scores,self.in_spacings,self.in_angles,self.in_phases=compute_scores_evo(
      self.inputs_flat,self.nx,self.L,num_steps=50)

          
    
      
  def gen_inputs_noisy_grids_scattered_fields(self,comp_scores=True):   
    """
    Generates regular grids and jitters the fields location
    """
    
    # parameter fitted by eye to match perfect-grid field size
    gau_sigma=self.grid_T/3.7
    seed(self.inputs_seed)
          
    # consider larger field for peaks in field but with center outside
    margin=self.grid_T
    XX,YY=np.mgrid[-self.L/2-margin:self.L/2+margin:self.dx,-self.L/2-margin:self.L/2+margin:self.dx]    

    # generate perfect grids  
    larger_pos=np.array([np.ravel(XX), np.ravel(YY)]).T   
    # generate perfect grids  
    seed(self.inputs_seed)
    
    self.phases,reg_grids,self.angle_vect,self.grid_T_vect=\
                                        gen_perfect_grids(larger_pos,self.n,self.grid_T,self.grid_angle,
                                        self.angle_sigma,jitter_axes_angle=self.jitter_axes_angle,
                                        grid_T_sigma=self.grid_T_sigma,jitter_axes_T=self.jitter_axes_T)
                                            
    
    inputs_large=np.zeros_like(reg_grids)     
    nx_large=int(np.sqrt(reg_grids.shape[0]))

    g_fun = lambda p: np.exp(-np.sum(p**2,2)/(2*gau_sigma**2))

    # add a gaussian for each noisy peak center      
    for grid_idx in xrange(self.N):
      grid=reg_grids[:,grid_idx].reshape(nx_large,nx_large)  
      noisy_peaks=get_noisy_peaks(grid,XX,YY,self.scatter_sigma)
      
      # gaussian input
      P0=larger_pos[np.newaxis,:,:]-noisy_peaks.T[:,np.newaxis,:]
      inputs_large[:,grid_idx]=g_fun(P0).astype(np.float32).sum(axis=0)
    
    
    # crop out the out the outer margin, we retain an inner square of size (self.nx, self.nx)
    margin_nx=np.int((nx_large-self.nx)/2.)
    inputs_unfolded=inputs_large.reshape(nx_large,nx_large,self.N)
    inputs=inputs_unfolded[margin_nx:margin_nx+self.nx,margin_nx:margin_nx+self.nx]     
    inputs=inputs.reshape(self.nx**2,self.N) 

    
    # shift down and clip (to mimic perfect grid inputs)
    inputs-=0.5
    inputs=inputs.clip(0,100)
    inputs=inputs/inputs.mean(axis=0)*self.input_mean
    
    self.inputs_flat=np.ascontiguousarray(inputs, dtype=np.float32)

    
    # compute scores
    if comp_scores:
      self.in_scores,self.in_spacings,self.in_angles,self.in_phases=compute_scores_evo(
      self.inputs_flat,self.nx,self.L,num_steps=50)
  
  

  def gen_inputs_bvc(self):
    """
    Generates Boundary Vector Cell inputs
    """
   
        
    d_ran=np.linspace(0.1,self.L/2.,num=self.n,endpoint=False)  
    phi_ran=np.linspace(0,2*np.pi,num=self.n,endpoint=False) 
    
    # standard deviation of the gaussian as a function of distance
    sigma_rad = lambda d: (d/self.beta+1)*self.sigma_rad_0
    
    # boundary vector field, i.e., the blob
    bvf= lambda p_dist,p_ang,d,phi: np.exp(-(p_dist-d)**2/(2*sigma_rad(d)**2))/(np.sqrt(2*np.pi)*sigma_rad(d)) *\
                                    np.exp(-(np.remainder((p_ang-phi),2*np.pi)-np.pi)**2/(2*self.sigma_ang**2))/(np.sqrt(2*np.pi)*self.sigma_ang)    
    
    # position of the walls
    east_wall=np.where(self.pos[:,0]==self.X.min())[0]
    west_wall=np.where(self.pos[:,0]==self.X.max())[0]
    north_wall=np.where(self.pos[:,1]==self.Y.max())[0]
    south_wall=np.where(self.pos[:,1]==self.Y.min())[0]
    wall_pos=np.hstack([east_wall,west_wall,north_wall,south_wall])
    num_walls=4
    
    pos_shift=self.pos[np.newaxis,:,:]
    p_wall_shift=self.pos[wall_pos,:][:,np.newaxis,:]-pos_shift
    p_wall_shift=p_wall_shift.reshape(self.nx*num_walls*self.nx**2,2)
    
    p_wall_dist=np.sqrt(np.sum(p_wall_shift**2,axis=1))
    p_wall_ang=np.arctan2(p_wall_shift[:,1],p_wall_shift[:,0])
    
    #p_dist=sqrt(np.sum(self.pos**2,axis=1))
    #p_ang=arctan2(self.pos[:,1],self.pos[:,0])
    

    self.inputs_flat=np.zeros((self.nx**2,self.N),dtype=np.float32)
    #self.blobs_flat=zeros((self.nx**2,self.N),dtype=float32)

    start_clock=clock()
    idx=0
    for d in d_ran:
      for phi in phi_ran:
        print_progress(idx,self.N,start_clock=start_clock)
        self.inputs_flat[:,idx]=np.mean(bvf(p_wall_dist,p_wall_ang,d,phi).reshape(self.nx*num_walls,self.nx**2),axis=0)
        #self.blobs_flat[:,idx]=bvf(p_dist,p_ang,d,phi)
        idx+=1
        
    # scale to fixed mean
    self.input_scalings=self.input_mean/np.mean(self.inputs_flat,axis=0)    
    self.inputs_flat*=self.input_scalings

      
  def gen_inputs_random(self):
    """
    Generate inputs from low-pass filtered noise
    """
    
    #dx=self.L/self.nx
    
    g_fun = lambda p:self.amp*np.exp(-np.sum(p**2,1)/(2*self.sigma**2))

    # white noise 

    # fixed amplitude
    if self.inputs_type == InputType.INPUT_GAU_MIX_POS_FIXAMP:
      wn=np.ones((self.nx,self.nx,self.N)).astype(np.float32)
    
    # only positive values
    if self.inputs_type in (InputType.INPUT_GAU_MIX_POS,InputType.INPUT_RAND,InputType.INPUT_RAND_CORR):
      #wn=abs(randn(self.nx,self.nx,self.N).astype(np.float32))
      wn=np.random.uniform(size=(self.nx,self.nx,self.N)).astype(np.float32)
      
    # positive and negative
    elif self.inputs_type in (InputType.INPUT_GAU_MIX,InputType.INPUT_GAU_MIX_NEG):
      wn=randn(self.nx,self.nx,self.N).astype(np.float32)

    if self.inputs_type in (InputType.INPUT_GAU_MIX,InputType.INPUT_GAU_MIX_NEG,
                            InputType.INPUT_GAU_MIX_POS,InputType.INPUT_GAU_MIX_POS_FIXAMP):

      assert(self.num_gau_mix<=self.nx**2)
      if self.num_gau_mix<self.nx**2:

        # set N-num_gau_mix elements to zero          
        wn_flat=wn.reshape(self.nx**2,self.N)

        self.random_amps=[]
        for i in xrange(self.N):        
          idxs_to_zero=permutation(self.nx**2)[:self.nx**2-self.num_gau_mix]
          non_zero_idxs=np.setdiff1d(np.arange(self.nx**2),idxs_to_zero)
          wn_flat[idxs_to_zero,i]=0
          self.random_amps.append(wn_flat[non_zero_idxs,i].tolist())
          
        wn=wn_flat.reshape(self.nx,self.nx,self.N).astype(np.float32)
                
    # convolution             
    wn_dft=fft2(wn,axes=[0,1]).astype(np.complex64)
    gx=g_fun(self.pos).reshape(self.nx,self.nx)
    filt_x=np.real(fftshift(ifft2(fft2(gx)[:,:,np.newaxis]*wn_dft,axes=[0,1]),axes=[0,1])).astype(np.float32)
    filt_xu=filt_x
    
    # no rescaling because noise had variance one (otherwise it should have variance 1/dx**2)
    #filt_xu*=dx**2
    
    # add Gaussian correlation across inputs    
    if self.inputs_type==InputType.INPUT_RAND_CORR:
      
      self.get_inputs_centers()
      gu=g_fun(self.centers).reshape(self.n,self.n)
      
      filt_x_dftu=fft2(filt_x.reshape(self.nx*self.nx,self.n,self.n),axes=[1,2]).astype(np.complex64)
      filt_xu=np.real(fftshift(ifft2(fft2(gu)[np.newaxis,:,:]*filt_x_dftu,axes=[1,2]),axes=[1,2])).astype(np.float32)
      #filt_xu*=dx**2
      
    # normalization
    filt_xu_flat=filt_xu.reshape(self.nx*self.nx,self.N)

    # adjust baseline by shifting the minimum at zero
    if self.inputs_type in (InputType.INPUT_GAU_MIX,InputType.INPUT_RAND,InputType.INPUT_RAND_CORR):
      filt_xu_flat-=np.amin(filt_xu_flat,axis=0)

    # shift to fixed mean    
    if self.inputs_type == InputType.INPUT_GAU_MIX_NEG:
      filt_xu_flat-=np.mean(filt_xu_flat,axis=0) 
      filt_xu_flat+=self.input_mean
      self.input_scalings=np.ones(self.N)
      
    else:
      # scale to fixed mean
      self.input_scalings=self.input_mean/np.mean(filt_xu_flat,axis=0)    
      filt_xu_flat*=self.input_scalings

    self.inputs_flat=np.ascontiguousarray(filt_xu_flat,dtype=np.float32)
            
  def gen_inputs_gaussian(self):
    """
    Generate Gaussian inputs
    """
    
    g_fun = lambda p: np.exp(-np.sum(p**2,2)/(2*self.sigma**2))*self.amp   

    # get input centers      
    self.get_inputs_centers()
    
    # gaussian input
    P0=self.pos[np.newaxis,:,:]-self.centers[:,np.newaxis,:]
    inputs=g_fun(P0).astype(np.float32)

      
    # add periodic boundaries    
    if self.periodic_inputs is True:
      for idx,center_shifted in enumerate(([0.,self.L], [0.,-self.L], [self.L,0.], [-self.L,0.], [-self.L,-self.L], [-self.L,self.L], [self.L,self.L], [self.L,-self.L])):
        P=P0+np.array(center_shifted)
        inputs+=g_fun(P).astype(np.float32)
              
    self.inputs_flat=np.ascontiguousarray(inputs.T.reshape(self.nx*self.nx,self.N), dtype=np.float32)
    
    

  def get_inputs_centers(self):
    """
    Computes input centers
    """
    
    if self.inputs_type==InputType.INPUT_GAU_RANDOM_CENTERS:
  
      # randomly distributed gaussian centers      
      SSX=(rand(self.n,self.n)-0.5)*self.L*self.outside_ratio
      SSY=(rand(self.n,self.n)-0.5)*self.L*self.outside_ratio
      self.centers= np.array([np.ravel(SSX), np.ravel(SSY)]).T
      
    else:

      # sample gaussian centers on a regular grid
      if self.periodic_inputs is True:
        ran,step=np.linspace(-self.L/2.,self.L/2.,self.n,endpoint=False,retstep=True)      
      else:
        
        ran,step=np.linspace(-self.L/2*self.outside_ratio,self.L/2*self.outside_ratio,self.n,endpoint=False,retstep=True)
        ran=ran+step/2.
        
      SSX,SSY = np.meshgrid(ran,ran)
      self.centers= np.array([np.ravel(SSX), np.ravel(SSY)]).T
    
      if self.inputs_type==InputType.INPUT_GAU_NOISY_CENTERS:
        NX=randn(self.n,self.n)*self.centers_std
        NY=randn(self.n,self.n)*self.centers_std
        
        self.centers+= np.array([np.ravel(NX), np.ravel(NY)]).T

  def plot_sample(self,random=False,num_samples=16):
    import pylab as pl
    from plotlib import noframe,colorbar
    from numpy import var,floor,ceil,arange,sqrt
    
    sparseness=self.comp_sparseness()

    if random is True:
      input_idxs=randint(0,self.n**2,num_samples)
    else:
      input_idxs=arange(num_samples)

    nsx=int(ceil(sqrt(num_samples)))
    nsy=int(floor(sqrt(num_samples)))
    pl.figure(figsize=(12,10))
    for idx,input_idx in enumerate(input_idxs):
      pl.subplot(nsx,nsy,idx+1,aspect='equal')
      noframe()
      pl.pcolormesh(self.inputs_flat[:,input_idx].reshape(self.nx,self.nx))
      colorbar()
      pl.title('m:%.3f v:%.2e s:%.2e'%(np.mean(self.inputs_flat[:,input_idx]),var(self.inputs_flat[:,input_idx]),sparseness[input_idx]),fontsize=14)
      
      
  def plot_scores(self):
    import pylab as pl
    from plotlib import custom_axes
    from numpy import median 
    from plotlib import MaxNLocator
    
    pl.figure(figsize=(2.8,1.8))
    pl.subplots_adjust(bottom=0.3,left=0.3)
    pl.hist(self.in_scores,bins=50,color='k')
    pl.axvline(median(self.in_scores),color='r')
    custom_axes()
    pl.xlabel('Input gridness score',fontsize=11)
    pl.ylabel('Number of neurons',fontsize=11)
    pl.xlim([-0.5,2])
    #pl.ylim([0,60])
    pl.gca().yaxis.set_major_locator(MaxNLocator(3))
    pl.gca().xaxis.set_major_locator(MaxNLocator(3))
      
  def get_overlap_mean(self,th=None):
    if th is None:
      th=self.amp/2.
    return np.mean(np.sum(self.inputs_flat>th,axis=1))


  def get_overlap_var(self,th=None):
    if th is None:
      th=self.amp/2.
    return np.var(np.sum(self.inputs_flat>th,axis=1))
    

  def get_overlap_cv(self,th=None):
    if th is None:
      th=self.amp/2.
    return np.sqrt(self.get_overlap_var(th))/self.get_overlap_mean(th)
    
  def get_num_overlp(self,th=None):  
    if th is None:
      th=self.amp/2.
    return np.sum(self.inputs_flat>th,axis=1).reshape(self.nx,self.nx)
    
  def plot_num_overlap(self,th=None):
    if th is None:
      th=self.amp/2.
    
    import pylab as pl
    import plotlib as pp

    pl.figure()
    pl.subplot(111,aspect='equal')
    pl.pcolormesh(np.sum(self.inputs_flat>th,axis=1).reshape(self.nx,self.nx))
    pp.noframe()
    pp.colorbar()    
      
  def get_var_of_mean_input(self)  :
    input_mean=self.inputs_flat.mean(axis=1).reshape(self.nx,self.nx)        
    return input_mean.var() 
      
  def get_cv_of_mean_input(self)  :
    input_mean=self.inputs_flat.mean(axis=1).reshape(self.nx,self.nx)        
    return np.sqrt(input_mean.var())/input_mean.mean() 
    
  def plot_mean_input(self) :
    import pylab as pl
    input_mean=self.inputs_flat.mean(axis=1).reshape(self.nx,self.nx)        
    self.dx=self.L/self.nx
    X,Y=np.mgrid[-self.L/2:self.L/2:self.dx,-self.L/2:self.L/2:self.dx]
  
    pl.figure()
    pl.subplot(111,aspect='equal')
    pl.pcolormesh(X,Y,input_mean)
    pl.title('var: %e'%input_mean.var())
    if self.inputs_type in (InputType.INPUT_GAU_GRID,
                            InputType.INPUT_GAU_NOISY_CENTERS,
                            InputType.INPUT_GAU_RANDOM_CENTERS):
      pl.plot(self.centers[:,0],self.centers[:,1],'.k')
    pl.colorbar()
    
  def get_boundary_diff(self):
    input_mean_beff=self.inputs_flat.mean(axis=1).reshape(self.nx,self.nx)        
    bound_diff=self.input_mean-input_mean_beff
    return bound_diff
  
  def plot_boundary_diff(self):
    import pylab as pl
    pl.figure()
    pl.subplot(111,aspect='equal')
    pl.pcolormesh(self.get_boundary_diff())
    pl.colorbar()
    
  def plot_corrected_total_input(self) :
    import pylab as pl
    input_mean=self.inputs_flat.mean(axis=1).reshape(self.nx,self.nx)        
    self.dx=self.L/self.nx
    X,Y=np.mgrid[-self.L/2:self.L/2:self.dx,-self.L/2:self.L/2:self.dx]
  
    pl.figure()
    pl.subplot(111,aspect='equal')
    pl.pcolormesh(X,Y,input_mean+self.get_boundary_diff())
    if self.inputs_type==InputType.INPUT_GAU_GRID:
      pl.plot(self.centers[:,0],self.centers[:,1],'.k')
    pl.colorbar()
    
  def comp_sparseness(self):
    return np.sum(self.inputs_flat,axis=0)**2/(self.nx**2*np.sum(self.inputs_flat**2,axis=0))
  
  def comp_mean(self):
    return np.mean(self.inputs_flat,axis=0)

  def comp_var(self):
    return np.var(self.inputs_flat,axis=0)
    
  def print_input_stats(self):
    print
    print 'N = %d' %self.n**2
    print 'L = %d' %self.L
    print 'InputType = %s'%self.inputs_type
    if self.inputs_type == InputType.INPUT_GAU_MIX_POS:
      print 'Num Gau Mix = %d'%self.num_gau_mix
    print    
    print 'Single-input mean = %.3f'%np.mean(self.comp_mean())
    print 'Single-input variance = %.3f'%np.mean(self.comp_var())
    print 'Single-input sparseness = %.3f'%np.mean(self.comp_sparseness())
    print 
    print 'Mean-input variance = %.e'%np.var(np.mean(self.inputs_flat,axis=1))
        


