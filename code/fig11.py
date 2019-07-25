# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 18:35:16 2017

@author: dalbis
"""



from grid_functions import g_ht_k
from simlib import ensureDir
import numpy as np
import pylab as pl

figures_path='../figures'
ensureDir(figures_path)

def get_filter_params(tau_in,tau_out,mu):
  b_in=1./tau_in
  b_out=1./tau_out
  
  g=b_out*(1+mu)
  c1=(b_in**2-b_in*b_out)/(b_in-g)
  c2=-b_in*b_out*mu/(b_in-g)
  return b_in,b_out,c1,c2,g
  


def get_max_frequency(tau_in,tau_out,mu,sigma=0.0625,speed=0.25,L=1.0,input_mean=0.4,N=900,ret_curve=False):
  
  
  fran=np.arange(0,5,0.001)

  
  b_in,b_out,c1,c2,g=get_filter_params(tau_in,tau_out,mu)
  
  fran=np.arange(0,5,0.001)

  
  amp=input_mean*L**2/(2*np.pi*sigma**2)
  
  
  K_ht_k_short= lambda k:(c1/speed)/(k**2+(b_in/speed)**2)**0.5 \
                       + (c2/speed)/(k**2+(g/speed)**2)**0.5 
  
  
  Kht_vect=K_ht_k_short(2*np.pi*fran)
  ght2_vect=g_ht_k(amp,sigma,2*np.pi*fran)**2
  corr_ht_est=Kht_vect*ght2_vect/L**2
  
  density=N/L**2
  raw_eigs=corr_ht_est*density
  
  
  max_freq=fran[raw_eigs.argmax()]
  mod_depth=raw_eigs.max()
  
  if ret_curve is True:
    return fran,raw_eigs,max_freq,mod_depth
  return max_freq,mod_depth
  
  
#%%

## COMPUTE MAX FREQUENCIES AND EIGENVALUES FOR A REGION OF THE PARAMETER SPACE

tau_in_def=0.01
tau_out_def=10.
def_a=0.
mu_def=100

tau_out_ran=np.arange(0.1,20.05,0.1)
mu_ran=np.arange(0,203,4)
  
tau_in_ran=np.arange(0.005,0.105,0.005)


max_freq_mat_tau_out=np.zeros((len(tau_out_ran),len(mu_ran)))
max_freq_mat_tau_in=np.zeros((len(tau_in_ran),len(mu_ran)))

max_eig_mat_tau_out=np.zeros_like(max_freq_mat_tau_out)
max_eig_mat_tau_in=np.zeros_like(max_freq_mat_tau_in)


for tau_out_idx,tau_out in enumerate(tau_out_ran):
  for mu_idx,mu in enumerate(mu_ran):
    max_freq,max_eig=get_max_frequency(tau_in_def,tau_out,mu)
    max_freq_mat_tau_out[tau_out_idx,mu_idx]=max_freq
    max_eig_mat_tau_out[tau_out_idx,mu_idx]=max_eig
      

for tau_in_idx,tau_in in enumerate(tau_in_ran):
  for mu_idx,mu in enumerate(mu_ran):
    max_freq,max_eig=get_max_frequency(tau_in,tau_out_def,mu)
    max_freq_mat_tau_in[tau_in_idx,mu_idx]=max_freq
    max_eig_mat_tau_in[tau_in_idx,mu_idx]=max_eig


      
    

#%%%%% PLOT MAX FREQUENCIES


def get_contour(CS,idx):
  path = CS.collections[idx].get_paths()[0]
  v = path.vertices
  x = v[:,0]
  y = v[:,1]
  return x,y,v

def interpolate_contour(CS,idx):
  x,y,v=get_contour(CS,idx)
  z = np.polyfit(x, y, 3)
  p = np.poly1d(z)
  v[:,1]=p(x)

import plotlib as pp
    
pl.rc('font',size=11)
pp.set_tick_size(4.5)



freq_levels=[1,2.,2.2,2.3,2.5] 
max_eig_levels=[0,1,5,10,20,30,40,50]


X_tau_out, Y_tau_out = np.meshgrid(tau_out_ran, -mu_ran)
X_tau_in, Y_tau_in = np.meshgrid(tau_in_ran, -mu_ran)


pl.figure(figsize=(8,6))
pl.subplots_adjust(bottom=0.2,wspace=0.3,hspace=0.5,right=0.9)



ax1=pl.subplot(2,2,1)


axes=[ax1]

# max_freq for tau_out
pl.sca(ax1)
CS_maxfreq_taul=pl.contourf(X_tau_out,Y_tau_out,max_freq_mat_tau_out.T,
                            levels=freq_levels,cmap='Greens')
                            
CS_maxfreq_taul_lines=pl.contour(X_tau_out,Y_tau_out,max_freq_mat_tau_out.T,
                                 levels=freq_levels,colors='k')




pl.clabel(CS_maxfreq_taul_lines, inline=0, fontsize=11,fmt='%.1f',colors='k')

pl.xlim(0.1,20)
pl.xticks([0.1,5,10,15,20])
pp.custom_axes()
pl.xlabel('Output kernel time constant t_out')
pl.ylabel('Output kernel integral')
pl.title('Critical spatial frequency [1/m]')


pl.savefig(figures_path+'/fig11.eps', dpi=300,transparent=True)









      
      
