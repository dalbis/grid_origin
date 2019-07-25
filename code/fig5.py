# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:13:55 2016

@author: dalbis
"""


import numpy as np
import pylab as pl
import plotlib as pp
import os
from grid_functions import load_data,compute_teo_eigs,compute_gaussian_teo_corr,map_merge
from grid_spikes import GridSpikes
from grid_params import GridSpikeParams
import gridlib as gl
from simlib import ensureDir

figures_path='../figures'
ensureDir(figures_path)


par_map=map_merge(GridSpikeParams.gau_grid_small_arena_biphasic_neg,{
                 'a':1.1,
                 'seed':30,
                 'variable_speed':False,
                  })
                
sim=GridSpikes(par_map)
sim.post_init()           


# generate results of the spiking model if not present
if not os.path.exists(sim.dataPath):
  sim.run()
  sim.post_run()
  
# load data
p,r=load_data(os.path.join(GridSpikes.results_path,'%s_data.npz'%sim.hash_id))


#%%
  
# reshaping
J_mat_det=r.J_vect.reshape(p.n,p.n,p.num_snaps)  


# time vector
num_sim_steps = int(p.sim_time/p.dt)
delta_snap = int(num_sim_steps/p.num_snaps)
snap_times=np.arange(p.num_snaps)*delta_snap*p.dt

# density of neurons
density=p.n**2/p.L**2

  
# time at which the first weight start saturating
time_first_sat_avg=None
if np.amin(r.J_vect)<p.eta and p.clip_weights is True:
  time_first_sat=snap_times[np.where(np.amin(r.J_vect,axis=0)<p.eta)[0][0]]


# DFTs and profiles 
num_J_dft_det,J_freqs_det,J_allfreqs_det=gl.dft2d_num(J_mat_det,p.L,p.n)
num_J_prof_det=gl.dft2d_profiles(num_J_dft_det)   

# teoretical eigenvalues (less accurate)
eigs_freqs,raw_eigs=compute_teo_eigs(None,r.paramMap[()],teo_input_pw=True)

samp_eigs_freqs=np.arange(0,p.nx/2,1/p.L)
samp_eigs_freqs,samp_raw_eigs=compute_teo_eigs(None,r.paramMap[()],teo_input_pw=True,freqs=samp_eigs_freqs)

# compute full eigenvalues
C_mean,C_mean_ft,corr_prof,corr_prof_ht,uran,fran,kran=compute_gaussian_teo_corr(True,
                                                                                 r.paramMap[()])

full_eigs=(C_mean_ft-p.a)*p.eta
  

J0_dft=np.fft.fftshift(np.fft.fft2(J_mat_det[:,:,0]))      
teo_full_J_dft=gl.dft2d_teo(J0_dft,full_eigs,snap_times,p.n)        
teo_full_J_prof=gl.dft2d_profiles(teo_full_J_dft)

print 'max_eig = %.2f'%(raw_eigs-p.a).max()
print 'tau_str= %.1e'%(1/(p.eta*(raw_eigs-p.a).max()))
  
#%%

# Number of weights at the lower bound (inset)

pl.rc('font',size=12)
pp.set_tick_size(4)

filt_len=5
num_at_bound=np.sum(r.J_vect<0.005,axis=0)/900.

frist_low_time_idx=np.argmin(abs(num_at_bound-0.25))
time_first_sat=snap_times[frist_low_time_idx]
print 't_sat=%.1e '%time_first_sat
time_first_sat=None


num_at_bound_filt=np.convolve(num_at_bound, np.ones((filt_len,))/filt_len, mode='valid')
pl.figure(figsize=(1.7,1.3))
pl.subplots_adjust(bottom=0.25,left=0.25)
pl.plot(snap_times[:len(num_at_bound_filt)],num_at_bound_filt,lw=2,color='k')

pp.custom_axes()
pl.gca().set_xscale('log')       
pl.yticks([0,0.5])
pl.xlabel('Time [s]')
pl.ylabel('Weights at lower bound')

pl.xlim(6e3 if pl.gca().get_xscale()=='log' else 0,p.sim_time)

pl.savefig(figures_path+'/fig5b_inset.eps',bbox_inches='tight',dpi=300)


#%%


### EVOLUTION 


snap_plot_idxs=np.array([0,10,20,40,100,199])

print [ '%.2e'%time for time in snap_times[snap_plot_idxs]]

fig=pl.figure(figsize=(9,3))
pl.subplots_adjust(hspace=0.05)
for pidx,idx in enumerate(snap_plot_idxs):
  pl.subplot(2,6,pidx+1,aspect='equal')
  mesh=pl.pcolormesh((J_mat_det[:,:,idx]),rasterized=True)
  pp.noframe()
  print 't = %.0e s'%snap_times[idx]
  print 'max =%.2f'%J_mat_det[:,:,idx].max()

cbar_ax = fig.add_axes([0.91, 0.54, 0.008, 0.14 ])    
pl.colorbar(mesh,cax=cbar_ax,ticks=[])
      
pl.rcParams['axes.linewidth'] = 0.5
for pidx,idx in enumerate(snap_plot_idxs):
  pl.subplot(2,6,pidx+7,aspect='equal')
  Mdft,mesh=pp.plot_matrixDFT(J_mat_det[:,:,idx],dt=p.L/p.n,circle_radius=3.,cmap='binary',circle_color=pp.red)
  pl.xticks([])
  pl.yticks([])

cbar_ax = fig.add_axes([0.91, 0.135, 0.008, 0.14 ])    
pl.colorbar(mesh,cax=cbar_ax,ticks=[])


pl.savefig(figures_path+'/fig5c.eps',bbox_inches='tight',dpi=300)


#%% distribution
logx=True

pl.rc('font',size=13)
pp.set_tick_size(4)

  
pl.figure(figsize=(8,2.5),facecolor='w')
pl.subplots_adjust(bottom=0.2,wspace=0.6,left=0.1,right=0.95)
pl.subplot(122)
np.random.seed(0)
idxs=np.random.randint(0,high=p.n**2,size=200)

pp.custom_axes()
pl.ylabel('Synaptic Weight')
pl.xlabel('Time [s]')

pl.xlim(snap_times[1] if pl.gca().get_xscale()=='log' else 0,p.sim_time)
if logx :
  pl.gca().set_xscale('log')        
  pl.xlim(6e3,1e6)

pl.ylim(0.,0.4)
pl.yticks([0,0.2,0.4])

if time_first_sat is not None:
  pl.axvline(time_first_sat,color='gray')


for snap_idx in snap_plot_idxs:
  pl.plot(snap_times[snap_idx],0.015,marker='v',mfc=pp.red,ms=10,mec='k')

#eigenvalues
pl.subplot(121)
pl.xlabel('Frequency [1/m]')
pl.ylabel('Eigenvalue')

pl.plot(eigs_freqs,raw_eigs-p.a,'-k')
pl.plot(samp_eigs_freqs,samp_raw_eigs-p.a,'ok',markersize=8,mfc=pp.red)
pl.axhline(0,color='k',linestyle='--')

pl.xlim(0.5,6.5)
pl.ylim(-6.5,2)
pl.yticks([-6,-4,-2,0,2])
pp.custom_axes()


  
pl.savefig(figures_path+'/fig5a.eps',bbox_inches='tight',dpi=300)

#%%

def plot_weight_dist(snap_times,J_vect,alpha=0.2,lines=True):
  if lines is True:
    pass
    pl.plot(snap_times,(J_vect.T),color=[.0,.0,.0,alpha],rasterized=True)
  else:
    nbins=100
    M=np.zeros((p.num_snaps,nbins))
    for idx in xrange(p.num_snaps):
      h,x=pl.histogram(J_vect[:,idx],bins=nbins,range=(0,0.4),normed=False)
      M[idx,:]=h/float(p.n**2)
    pl.pcolormesh(snap_times,np.linspace(0,0.4,nbins),M.T,cmap='binary',vmax=0.02,vmin=0)
    pl.gca().set_xscale('log')        
    return M
    
pl.figure(figsize=(8,2.5),facecolor='w')
pl.subplots_adjust(bottom=0.2,wspace=0.6,left=0.1,right=0.95)
pl.subplot(122,rasterized=True)
np.random.seed(0)
idxs=np.random.randint(0,high=p.n**2,size=200)
M=plot_weight_dist(snap_times,r.J_vect[:,:],alpha=0.2,lines=False)

pl.axis('off')
pl.ylim(0.,0.4)
pl.xlim(6e3,1e6)

pl.savefig(figures_path+'/fig5b.eps', dpi=300,transparent=True)





#%% plot profiles

pl.rcParams['axes.linewidth'] = 1

fig=pl.figure(figsize=(10,7),facecolor='white')

pl.subplots_adjust(left=0.08,right=0.98,wspace=0.4,hspace=0.45,bottom=0.1,top=0.93)  
logy=True
logx=True
ymin=np.amin(num_J_prof_det) 
ymax=np.amax(num_J_prof_det)
ymin=1e-6 if ymin ==0 and logy is True else ymin
  
freq_idxs=(2,3,6,7)

for idx,freq_idx in enumerate(freq_idxs):
  pl.subplot(2,2,idx+1)
  pp.custom_axes()

  pl.plot(snap_times,teo_full_J_prof[:,freq_idx],color=pp.red,linewidth=2)
    
  # plot numerical
  pl.plot(snap_times,num_J_prof_det[:,freq_idx],'-k',linewidth=1.5)
  



  if logy : pl.gca().set_yscale('log')
  if logx : pl.gca().set_xscale('log')
    
  pl.ylim(ymin,ymax)
  pl.xlim(1e3 if pl.gca().get_xscale()=='log' else 0,p.sim_time)
  
  pl.title('%.2f [1/m] '%samp_eigs_freqs[freq_idx])

  if time_first_sat is not None: pl.axvline(time_first_sat,color='k',linestyle=':',linewidth=1.5)

  if freq_idx in (1,3):     
    pl.ylabel('Amplitude')


  if freq_idx>3:     
    pl.xlabel('Time [s]')


logx=True
logy=False
freq_idx=3
figw=pl.figure(figsize=(.7,.7))
wax=pl.subplot(111,projection='polar')



pl.rc('font',size=13)
pp.set_tick_size(4)

pl.figure(figsize=(8,2.5),facecolor='w')
pl.subplot(1,2,1)
pl.subplots_adjust(bottom=0.2,wspace=0.6,left=0.1,right=0.95)

angles,amps=gl.get_angle_amps(num_J_dft_det,freq_idx,p.n)
pp.plot_dft_angles(snap_times,angles,amps,p.sim_time,plot_wheel=True,wheel_axis=wax,cmap='hsv',lw=2)

for snap_idx in snap_plot_idxs[2:]:
  pl.plot(snap_times[snap_idx],0.6,marker='v',mfc='k',ms=9,mec='none')
    
if time_first_sat is not None:
  pl.axvline(time_first_sat,color='gray',linewidth=1.5)

if logx : pl.gca().set_xscale('log')
if logy : pl.gca().set_yscale('log')
pl.xlim(1e5 if pl.gca().get_xscale()=='log' else 0,p.sim_time)
pl.ylabel('Amplitude at 0.75 [1/m]')
pl.xlabel('Time [s]')
pl.xticks([1e5,2e5,5e5,1e6],['','','',''])  
pl.yticks([0,10,20,30])


pl.subplot(1,2,2)
pl.plot(snap_times,r.scores,'-k',linewidth=1.5)
for snap_idx in snap_plot_idxs[2:]:
  pl.plot(snap_times[snap_idx],-0.45,marker='v',mfc='k',ms=9,mec='none')
pp.custom_axes()
if logx : pl.gca().set_xscale('log')
if time_first_sat is not None:
  pl.axvline(time_first_sat,color='gray',linewidth=1.5)

pl.xlim(1e5 if pl.gca().get_xscale()=='log' else 0,p.sim_time)

pl.xlabel('Time [s]')
pl.ylabel('Gridness score')

pl.xticks([1e5,2e5,5e5,1e6],['','','',''])  



pl.savefig(figures_path+'/fig5de.eps', dpi=300,transparent=True)

pl.figure(figw.number)
pl.savefig(figures_path+'/fig5d_wheel.eps', dpi=300,transparent=True)